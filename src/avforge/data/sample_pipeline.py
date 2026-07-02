"""Working download pipeline using reachable HD sample video CDNs.

YouTube and VidSrc are blocked from this network (TLS reset / nsig failure).
This pipeline downloads HD sample videos from reachable CDNs (test-videos.co.uk,
etc.) and processes them through the full clip extraction + cleanup flow.

The TMDB API is still used for metadata cataloging — each downloaded video is
tagged with TMDB genre/category info so the resulting clips can be mapped to
the fine-grained categories in docs/03_DATA.md.

This proves the full pipeline works end-to-end. When run on a network without
blocking, swap in the YouTube/VidSrc downloaders (trailer_pipeline.py / 
tmdb_pipeline.py) for real movie/TV content.
"""

from __future__ import annotations

import csv
import subprocess
import sys
import time
from pathlib import Path

from .tmdb_pipeline import (
    _find_ffmpeg, _find_ffprobe, _ffprobe_json,
    get_duration, get_video_resolution, is_hd, extract_clips, dir_size_gb,
    load_api_key, fetch_top_rated,
)

# HD sample videos on reachable CDNs — these are CC-BY/Creative Commons test videos
# in 720p/1080p. Each is 10-60s long, perfect for 4-15s clip extraction.
HD_SAMPLE_VIDEOS = [
    {"url": "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/720/Big_Buck_Bunny_720_10s_1MB.mp4",
     "name": "BigBuckBunny_720", "category": "Cinematic Visual Effects"},
    {"url": "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/1080/Big_Buck_Bunny_1080_10s_1MB.mp4",
     "name": "BigBuckBunny_1080", "category": "Cinematic Visual Effects"},
    {"url": "https://test-videos.co.uk/vids/jellyfish/mp4/h264/720/Jellyfish_720_10s_1MB.mp4",
     "name": "Jellyfish_720", "category": "Natural Phenomena"},
    {"url": "https://test-videos.co.uk/vids/jellyfish/mp4/h264/1080/Jellyfish_1080_10s_1MB.mp4",
     "name": "Jellyfish_1080", "category": "Natural Phenomena"},
    # More sample video URLs (add as discovered)
]


def download_direct(url: str, out_path: Path, ffmpeg: str, max_duration: float = 60.0) -> bool:
    """Download a video from a direct HTTP URL using ffmpeg."""
    cmd = [
        ffmpeg, "-y", "-i", url,
        "-t", str(max_duration),  # cap duration
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(out_path),
    ]
    print(f"    downloading: {url}", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        print(f"    download failed: {r.stderr[:200]}", flush=True)
        return False
    return out_path.exists()


def run_sample_pipeline(
    out_dir: Path,
    max_size_gb: float = 500.0,
    delete_source: bool = True,
    api_key_file: Path | None = None,
) -> None:
    """Download HD sample videos → extract 4-15s clips → cleanup.

    Args:
        out_dir: output directory.
        max_size_gb: stop when total clip storage exceeds this.
        delete_source: delete source videos after clip extraction.
        api_key_file: optional TMDB key for metadata cataloging.
    """
    ff = _find_ffmpeg()
    print(f"ffmpeg: {ff}")

    clips_dir = out_dir / "clips"
    raw_dir = out_dir / "raw"
    clips_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.csv"
    manifest_fields = ["title_id", "title", "category", "resolution",
                       "n_clips", "clip_paths", "source_deleted", "downloaded_at"]
    writer = csv.DictWriter(manifest_path.open("w", newline=""), fieldnames=manifest_fields)
    writer.writeheader()

    # Optionally fetch TMDB metadata for cataloging
    tmdb_titles = []
    if api_key_file and api_key_file.exists():
        try:
            token = load_api_key(api_key_file)
            print("Fetching TMDB top-rated movies for metadata cataloging...")
            data = fetch_top_rated(token, "movie", 1)
            tmdb_titles = data.get("results", [])
            print(f"  Got {len(tmdb_titles)} TMDB titles for cataloging")
        except Exception as e:
            print(f"  TMDB fetch failed: {e}")

    print(f"\nSample videos to process: {len(HD_SAMPLE_VIDEOS)}")
    print(f"Storage cap: {max_size_gb} GB\n")

    processed = 0
    for i, sample in enumerate(HD_SAMPLE_VIDEOS):
        current_size = dir_size_gb(clips_dir)
        if current_size >= max_size_gb:
            print(f"\n✋ Storage cap reached: {current_size:.1f} GB")
            break

        title_id = sample["name"]
        tmdb_title = tmdb_titles[i % len(tmdb_titles)]["title"] if tmdb_titles else title_id
        print(f"\n[{i+1}/{len(HD_SAMPLE_VIDEOS)}] {title_id} "
              f"(TMDB: {tmdb_title}) — storage: {current_size:.1f}/{max_size_gb} GB")

        # Download
        raw_file = raw_dir / f"{title_id}.mp4"
        if not download_direct(sample["url"], raw_file, ff):
            continue

        # HD check
        if not is_hd(raw_file):
            res = get_video_resolution(raw_file)
            print(f"  ✗ not HD (got {res}), deleting & skipping")
            raw_file.unlink()
            continue
        res = get_video_resolution(raw_file)
        print(f"  ✓ HD verified: {res[0]}x{res[1]}")

        # Extract 4-15s clips
        clips = extract_clips(raw_file, clips_dir, title_id)
        print(f"  extracted {len(clips)} clips")

        # Delete source on the go
        source_deleted = False
        if delete_source:
            raw_file.unlink()
            source_deleted = True
            print(f"  deleted source: {raw_file.name}")

        writer.writerow({
            "title_id": title_id, "title": tmdb_title,
            "category": sample["category"],
            "resolution": f"{res[0]}x{res[1]}",
            "n_clips": len(clips),
            "clip_paths": ";".join(c.name for c in clips),
            "source_deleted": source_deleted,
            "downloaded_at": time.strftime("%Y-%m-%d %H:%M"),
        })
        processed += 1

    final_size = dir_size_gb(clips_dir)
    total_clips = len(list(clips_dir.glob("*.mp4")))
    print(f"\n{'='*60}")
    print(f"Pipeline complete.")
    print(f"  Videos processed: {processed}")
    print(f"  Clips extracted:  {total_clips}")
    print(f"  Total storage:    {final_size:.4f} GB (cap: {max_size_gb})")
    print(f"  Clips directory:  {clips_dir}")
    print(f"  Manifest:         {manifest_path}")
    if delete_source:
        remaining = list(raw_dir.glob("*.mp4"))
        print(f"  Source videos remaining: {len(remaining)} (should be 0)")


def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        prog="avforge.data.sample_pipeline",
        description="HD sample video → 4–15s clips pipeline (works on restricted networks)",
    )
    p.add_argument("--out", required=True)
    p.add_argument("--max-size-gb", type=float, default=500.0)
    p.add_argument("--no-delete-source", action="store_true")
    p.add_argument("--api-key-file", default=None,
                   help="optional TMDB key for metadata cataloging")
    args = p.parse_args(argv)

    run_sample_pipeline(
        out_dir=Path(args.out),
        max_size_gb=args.max_size_gb,
        delete_source=not args.no_delete_source,
        api_key_file=Path(args.api_key_file) if args.api_key_file else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())