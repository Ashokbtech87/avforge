"""TMDB trailer-based download pipeline.

VidSrc is blocked from this network (TLS connection reset). This module uses
TMDB's /movie/{id}/videos and /tv/{id}/videos endpoints to find official
YouTube trailers, then downloads them with yt-dlp (which works fine with
YouTube). Trailers are HD, high-quality, and legally available.

This is a practical alternative to VidSrc for collecting training data:
  - TMDB provides YouTube trailer IDs for most movies/TV series
  - yt-dlp handles YouTube downloads without TLS issues
  - Trailers are typically 1-3 minutes → multiple 4-15s clips each
  - HD quality is available (720p/1080p)
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from .tmdb_pipeline import (
    TMDB_API_BASE, CANONICAL_FPS, CANONICAL_RES, CANONICAL_AUDIO_SR,
    CANONICAL_AUDIO_CH, CLIP_MIN_S, CLIP_MAX_S, CLIP_STEP_S,
    HD_MIN_WIDTH, HD_MIN_HEIGHT,
    _find_ffmpeg, _find_ffprobe, _ffprobe_json,
    load_api_key, tmdb_get, fetch_top_rated, fetch_trending, fetch_popular,
    get_duration, get_video_resolution, is_hd, extract_clips, dir_size_gb,
)
import csv


def fetch_videos(token: str, media_type: str, tmdb_id: int) -> dict:
    """Fetch trailers/teasers for a movie or TV series from TMDB."""
    return tmdb_get(f"{media_type}/{tmdb_id}/videos", token)


def find_best_trailer(videos_data: dict) -> str | None:
    """Find the best YouTube trailer URL from TMDB videos response.

    Prefers: official Trailer > official Teaser > any Trailer > any video.
    Returns a YouTube URL or None.
    """
    results = videos_data.get("results", [])
    if not results:
        return None

    # Priority: official trailer on YouTube > teaser > clip > any
    priority = ["Trailer", "Teaser", "Clip", "Featurette", "Behind the Scenes"]
    youtube_videos = [v for v in results if v.get("site") == "YouTube" and v.get("key")]

    for ptype in priority:
        for v in youtube_videos:
            if v.get("type") == ptype and v.get("official"):
                return f"https://www.youtube.com/watch?v={v['key']}"
    # Fallback: any YouTube video
    for ptype in priority:
        for v in youtube_videos:
            if v.get("type") == ptype:
                return f"https://www.youtube.com/watch?v={v['key']}"
    if youtube_videos:
        v = youtube_videos[0]
        return f"https://www.youtube.com/watch?v={v['key']}"
    return None


def download_youtube(url: str, out_path: Path, resolution: str = "720") -> bool:
    """Download a YouTube video using yt-dlp."""
    # Use simple format strings that work even when nsig extraction partially fails.
    # "best" always works; height-filtered formats may fail if nsig blocks some streams.
    fmt_map = {
        "720": "best[height<=720]/best",
        "1080": "best[height<=1080]/best",
        "best": "best",
    }
    cmd = [
        "yt-dlp", "--no-warnings", "--no-playlist",
        "-f", fmt_map.get(resolution, fmt_map["best"]),
        "--merge-output-format", "mp4",
        "--no-check-certificates",
        "-o", str(out_path),
        url,
    ]
    print(f"    yt-dlp downloading: {url}", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        # Fallback: try plain "best" format
        print(f"    retrying with best format...", flush=True)
        cmd_fallback = [
            "yt-dlp", "--no-warnings", "--no-playlist",
            "-f", "best",
            "--merge-output-format", "mp4",
            "-o", str(out_path),
            url,
        ]
        r2 = subprocess.run(cmd_fallback, capture_output=True, text=True, timeout=600)
        if r2.returncode != 0:
            print(f"    yt-dlp failed: {r2.stderr[:300]}", flush=True)
            return False
    return out_path.exists()


def run_trailer_pipeline(
    api_key_file: Path,
    out_dir: Path,
    max_size_gb: float = 500.0,
    max_titles: int = 500,
    resolution: str = "720",
    media_types: tuple[str, ...] = ("movie", "tv"),
    delete_source: bool = True,
    min_rating: float = 8.0,
) -> None:
    """TMDB trailers → download → 4-15s clips → cleanup pipeline.

    Uses TMDB /videos endpoint to find YouTube trailers, downloads with yt-dlp,
    extracts 4-15s clips with ffmpeg, deletes source on the go.
    """
    token = load_api_key(api_key_file)
    print(f"TMDB API token loaded from {api_key_file}")

    ff = _find_ffmpeg()
    print(f"ffmpeg: {ff}")

    if not shutil.which("yt-dlp"):
        print("ERROR: yt-dlp not found. Run: pip install yt-dlp", file=sys.stderr)
        sys.exit(1)
    print(f"Download backend: yt-dlp (YouTube trailers via TMDB)")

    clips_dir = out_dir / "clips"
    raw_dir = out_dir / "raw"
    clips_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.csv"
    manifest_fields = ["title_id", "title", "media_type", "tmdb_id", "year",
                       "rating", "trailer_url", "n_clips", "clip_paths",
                       "source_deleted", "downloaded_at"]
    writer = csv.DictWriter(manifest_path.open("w", newline=""), fieldnames=manifest_fields)
    writer.writeheader()

    # --- Fetch high-rated titles from TMDB ---
    titles: list[dict] = []
    for media_type in media_types:
        for source_name, fetch_fn in [
            ("top_rated", fetch_top_rated),
            ("trending", fetch_trending),
            ("popular", fetch_popular),
        ]:
            if len(titles) >= max_titles * 3:
                break
            print(f"\nFetching {source_name} {media_type} (min rating: {min_rating})...")
            for page in range(1, 11):
                try:
                    if source_name == "trending":
                        data = fetch_fn(token, media_type, "week", page)
                    else:
                        data = fetch_fn(token, media_type, page)
                    for item in data.get("results", []):
                        vote = item.get("vote_average", 0)
                        if vote < min_rating:
                            continue
                        titles.append({
                            "media_type": media_type,
                            "tmdb_id": item["id"],
                            "title": item.get("title") or item.get("name", ""),
                            "year": (item.get("release_date") or item.get("first_air_date", ""))[:4],
                            "rating": vote,
                        })
                except Exception as e:
                    print(f"  page {page}: {e}")
                    break
                if len(titles) >= max_titles * 3:
                    break

    # Dedupe
    seen = set()
    deduped = []
    for t in titles:
        key = (t["media_type"], t["tmdb_id"])
        if key not in seen:
            seen.add(key)
            deduped.append(t)
    titles = deduped[:max_titles]

    print(f"\nTotal titles: {len(titles)} (rating ≥ {min_rating})")
    print(f"Storage cap: {max_size_gb} GB\n")

    # --- Process each title ---
    processed = 0
    skipped = 0
    for title in titles:
        current_size = dir_size_gb(clips_dir)
        if current_size >= max_size_gb:
            print(f"\n✋ Storage cap reached: {current_size:.1f} GB ≥ {max_size_gb} GB")
            break

        title_id = f"{title['media_type']}_{title['tmdb_id']}"
        print(f"\n[{processed+skipped+1}/{len(titles)}] {title['title']} ({title['year']}) "
              f"⭐{title['rating']}/10 — storage: {current_size:.1f}/{max_size_gb} GB")

        # Fetch trailer URL from TMDB
        try:
            videos = fetch_videos(token, title["media_type"], title["tmdb_id"])
        except Exception as e:
            print(f"  TMDB videos fetch failed: {e}")
            skipped += 1
            continue

        trailer_url = find_best_trailer(videos)
        if not trailer_url:
            print(f"  no YouTube trailer found, skipping")
            skipped += 1
            writer.writerow({
                "title_id": title_id, "title": title["title"],
                "media_type": title["media_type"], "tmdb_id": title["tmdb_id"],
                "year": title["year"], "rating": title["rating"],
                "trailer_url": "", "n_clips": 0, "clip_paths": "",
                "source_deleted": False, "downloaded_at": "",
            })
            continue

        print(f"  trailer: {trailer_url}")

        # Download trailer
        raw_file = raw_dir / f"{title_id}.mp4"
        if not download_youtube(trailer_url, raw_file, resolution):
            print(f"  download failed, skipping")
            skipped += 1
            continue

        # HD check
        if not is_hd(raw_file):
            res = get_video_resolution(raw_file)
            print(f"  ✗ not HD (got {res}), deleting & skipping")
            raw_file.unlink()
            skipped += 1
            continue
        print(f"  ✓ HD verified: {get_video_resolution(raw_file)}")

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
            "title_id": title_id, "title": title["title"],
            "media_type": title["media_type"], "tmdb_id": title["tmdb_id"],
            "year": title["year"], "rating": title["rating"],
            "trailer_url": trailer_url, "n_clips": len(clips),
            "clip_paths": ";".join(c.name for c in clips),
            "source_deleted": source_deleted,
            "downloaded_at": time.strftime("%Y-%m-%d %H:%M"),
        })
        processed += 1

    final_size = dir_size_gb(clips_dir)
    total_clips = len(list(clips_dir.glob("*.mp4")))
    print(f"\n{'='*60}")
    print(f"Pipeline complete.")
    print(f"  Titles processed: {processed} (skipped: {skipped})")
    print(f"  Clips extracted:  {total_clips}")
    print(f"  Total storage:    {final_size:.1f} GB (cap: {max_size_gb})")
    print(f"  Clips directory:  {clips_dir}")
    print(f"  Manifest:         {manifest_path}")


import shutil  # used in run_trailer_pipeline


def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        prog="avforge.data.trailer_pipeline",
        description="TMDB trailers → yt-dlp → 4–15s clips pipeline (HD, rating ≥ 8.0)",
    )
    p.add_argument("--api-key-file", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-size-gb", type=float, default=500.0)
    p.add_argument("--max-titles", type=int, default=500)
    p.add_argument("--resolution", choices=["720", "1080", "best"], default="720")
    p.add_argument("--media-types", nargs="+", default=["movie", "tv"], choices=["movie", "tv"])
    p.add_argument("--no-delete-source", action="store_true")
    p.add_argument("--min-rating", type=float, default=8.0)
    args = p.parse_args(argv)

    run_trailer_pipeline(
        api_key_file=Path(args.api_key_file),
        out_dir=Path(args.out),
        max_size_gb=args.max_size_gb,
        max_titles=args.max_titles,
        resolution=args.resolution,
        media_types=tuple(args.media_types),
        delete_source=not args.no_delete_source,
        min_rating=args.min_rating,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())