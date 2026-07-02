"""TMDB + Streambert/vid-dl-cli download → 4–15s clip pipeline for AV-Forge 2.0.

This module implements the data collection workflow from docs/06_DATA_COLLECTION.md
using:
  - TMDB API (https://api.themoviedb.org/3) for movie/TV metadata + stream URLs
  - vid-dl-cli-only (https://github.com/truelockmc/vid-dl-cli-only) as the download
    backend (a yt-dlp wrapper that handles m3u8/HLS streams from VidSrc etc.)
  - ffmpeg for clip extraction (4–15s segments at 720p/24fps)

Workflow per title:
  1. Query TMDB for trending/popular movies + TV series.
  2. For each title, construct a VidSrc stream URL (the source Streambert uses).
  3. Download the full video via vid-dl-cli (or yt-dlp directly as fallback).
  4. Extract multiple 4–15s clips from the downloaded video using ffmpeg.
  5. Delete the full source video immediately after clip extraction (on the go).
  6. Track total clip storage; stop when the 500 GB cap is reached.

Usage:
  python -m avforge.data.tmdb_pipeline --api-key-file D:\\AIModels\\TMDB_API.txt \\
      --out data/tmdb_clips --max-size-gb 500 --max-titles 500

Prerequisites:
  - ffmpeg + ffprobe installed and on PATH
  - vid-dl-cli-only downloaded from https://github.com/truelockmc/vid-dl-cli-only/releases
    (or yt-dlp installed: pip install yt-dlp)
  - TMDB API key (read access token)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TMDB_API_BASE = "https://api.themoviedb.org/3"
TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w500"
# VidSrc stream URL patterns (the source Streambert uses for streaming/downloading)
VIDSRC_MOVIE_URL = "https://vidsrc.to/embed/movie/{id}"
VIDSRC_TV_URL = "https://vidsrc.to/embed/tv/{id}/{season}/{episode}"

# Clip extraction parameters (from docs/06_DATA_COLLECTION.md + 03_DATA.md)
CLIP_MIN_S = 4.0
CLIP_MAX_S = 15.0
CLIP_STEP_S = 10.0       # extract a clip every ~10s of source video
CANONICAL_FPS = 24
CANONICAL_RES = (1280, 720)  # 720p (HD minimum)
CANONICAL_AUDIO_SR = 24000
CANONICAL_AUDIO_CH = 2

# HD-only policy: reject anything below 720p (1280x720)
HD_MIN_WIDTH = 1280
HD_MIN_HEIGHT = 720
HD_RESOLUTIONS = ("720", "1080", "best")  # allowed download resolutions


# ---------------------------------------------------------------------------
# TMDB API client
# ---------------------------------------------------------------------------

def load_api_key(key_file: Path) -> str:
    """Parse the TMDB API file. Supports both 'API <key>' and 'API read access token <jwt>'."""
    text = key_file.read_text().strip()
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("eyJ"):  # JWT read access token
            return line
    # Fallback: look for the API key (v3 auth)
    for line in text.splitlines():
        line = line.strip()
        if line and not line.lower().startswith("api read") and not line.lower().startswith("api\n"):
            if len(line) == 32 and line.isalnum():
                return line
    # Just return the first non-empty, non-header line
    for line in text.splitlines():
        line = line.strip()
        if line and "api" not in line.lower()[:4]:
            return line
    raise ValueError(f"Could not parse API key from {key_file}")


def tmdb_get(endpoint: str, token: str, params: dict | None = None) -> dict:
    """Make an authenticated TMDB API v3 request using the read access token."""
    import urllib.request
    import urllib.parse

    url = f"{TMDB_API_BASE}/{endpoint.lstrip('/')}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    })
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def fetch_trending(token: str, media_type: str = "movie", window: str = "week",
                   page: int = 1) -> dict:
    """Fetch trending movies or TV series from TMDB."""
    return tmdb_get(f"trending/{media_type}/{window}", token, {"page": page})


def fetch_popular(token: str, media_type: str = "movie", page: int = 1) -> dict:
    """Fetch popular movies or TV series."""
    return tmdb_get(f"{media_type}/popular", token, {"page": page})


def fetch_top_rated(token: str, media_type: str = "movie", page: int = 1) -> dict:
    """Fetch top-rated movies or TV series."""
    return tmdb_get(f"{media_type}/top_rated", token, {"page": page})


def fetch_tv_season(token: str, tv_id: int, season: int) -> dict:
    """Fetch episodes for a TV season."""
    return tmdb_get(f"tv/{tv_id}/season/{season}", token)


def fetch_title_details(token: str, media_type: str, tmdb_id: int) -> dict:
    """Fetch full details for a movie or TV series."""
    return tmdb_get(f"{media_type}/{tmdb_id}", token)


# ---------------------------------------------------------------------------
# ffmpeg/ffprobe locator (uses imageio_ffmpeg as fallback)
# ---------------------------------------------------------------------------

_FFMPEG: str | None = None
_FFPROBE: str | None = None


def _project_bin() -> Path:
    """Return the project root directory (where ffmpeg.exe/ffprobe.exe may live)."""
    # src/avforge/data/tmdb_pipeline.py → project root is 3 levels up
    return Path(__file__).resolve().parents[3]


def _find_ffmpeg() -> str:
    """Find ffmpeg executable — on PATH, in project dir, or via imageio_ffmpeg."""
    global _FFMPEG
    if _FFMPEG:
        return _FFMPEG
    # 1. On system PATH
    path = shutil.which("ffmpeg")
    if path:
        _FFMPEG = path
        return path
    # 2. In the project root directory (e.g. ffmpeg.exe next to pyproject.toml)
    candidate = _project_bin() / "ffmpeg.exe"
    if candidate.exists():
        _FFMPEG = str(candidate)
        return _FFMPEG
    candidate = _project_bin() / "ffmpeg"
    if candidate.exists():
        _FFMPEG = str(candidate)
        return _FFMPEG
    # 3. Via imageio_ffmpeg package
    try:
        import imageio_ffmpeg
        _FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
        return _FFMPEG
    except ImportError:
        pass
    raise RuntimeError("ffmpeg not found. Install it on PATH, put ffmpeg.exe in "
                       "the project root, or run: pip install imageio-ffmpeg")


def _find_ffprobe() -> str:
    """Find ffprobe — on PATH, in project dir, or derive from ffmpeg path."""
    global _FFPROBE
    if _FFPROBE:
        return _FFPROBE
    # 1. On system PATH
    path = shutil.which("ffprobe")
    if path:
        _FFPROBE = path
        return path
    # 2. In the project root directory
    candidate = _project_bin() / "ffprobe.exe"
    if candidate.exists():
        _FFPROBE = str(candidate)
        return _FFPROBE
    candidate = _project_bin() / "ffprobe"
    if candidate.exists():
        _FFPROBE = str(candidate)
        return _FFPROBE
    # 3. Derive from the ffmpeg binary path
    ff = _find_ffmpeg()
    probe_candidate = ff.replace("ffmpeg", "ffprobe")
    if Path(probe_candidate).exists():
        _FFPROBE = probe_candidate
        return _FFPROBE
    # No ffprobe available — we'll use ffmpeg -i for probing instead.
    _FFPROBE = ""  # sentinel: use ffmpeg-based probing
    return _FFPROBE


def _ffprobe_json(path: Path) -> dict:
    """Run ffprobe and return parsed JSON. Falls back to ffmpeg -i parsing."""
    probe = _find_ffprobe()
    if probe:
        r = subprocess.run(
            [probe, "-v", "quiet", "-print_format", "json",
             "-show_format", "-show_streams", str(path)],
            capture_output=True, text=True,
        )
        try:
            return json.loads(r.stdout)
        except json.JSONDecodeError:
            pass
    # Fallback: use ffmpeg -i and parse stderr for duration + resolution
    ff = _find_ffmpeg()
    r = subprocess.run([ff, "-i", str(path)], capture_output=True, text=True)
    err = r.stderr
    info = {"format": {}, "streams": [{}]}
    # parse Duration: 00:01:23.45
    for line in err.splitlines():
        if "Duration:" in line:
            parts = line.split("Duration:")[1].split(",")[0].strip()
            h, m, s = parts.split(":")
            info["format"]["duration"] = str(float(h) * 3600 + float(m) * 60 + float(s))
        if "Stream #0:0" in line and "Video:" in line:
            # parse resolution: e.g. "1920x1080"
            for token in line.split():
                if "x" in token and token[0].isdigit():
                    w, h = token.split("x")
                    info["streams"][0]["width"] = int(w)
                    info["streams"][0]["height"] = int(h.rstrip(","))
    return info


# ---------------------------------------------------------------------------
# Download backend (vid-dl-cli-only or yt-dlp fallback)
# ---------------------------------------------------------------------------

def find_vid_dl_cli() -> str | None:
    """Find the vid-dl-cli-only executable."""
    # Check common locations
    candidates = [
        "vid-dl-cli",
        "video-downloader-cli",
        "video-downloader",
    ]
    for c in candidates:
        path = shutil.which(c)
        if path:
            return path
    # Check for yt-dlp as fallback
    if shutil.which("yt-dlp"):
        return "yt-dlp"
    return None


def download_video(url: str, out_path: Path, downloader: str,
                   resolution: str = "720") -> bool:
    """Download a video using vid-dl-cli-only or yt-dlp.

    Args:
        url: the stream URL (VidSrc embed or direct m3u8).
        out_path: target file path (without extension for vid-dl-cli).
        downloader: path to the downloader executable.
        resolution: max height ("720", "1080", "best").
    Returns:
        True on success.
    """
    out_dir = str(out_path.parent)
    if "yt-dlp" in downloader:
        # yt-dlp fallback: download best mp4 up to 720p
        cmd = [
            downloader, "-f", f"bestvideo[height<={resolution}]+bestaudio/best[height<={resolution}]",
            "--merge-output-format", "mp4",
            "-o", str(out_path.with_suffix(".mp4")),
            "--no-playlist", "--no-warnings",
            url,
        ]
    else:
        # vid-dl-cli-only
        cmd = [
            downloader, url,
            "--format", "mp4 (with Audio)",
            "--resolution", resolution,
            "--folder", out_dir,
            "--filename", out_path.stem,
        ]
    print(f"    downloading: {' '.join(cmd[:6])}...", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        print(f"    download failed: {r.stderr[:300]}", flush=True)
        return False
    # Find the actual output file (vid-dl-cli may append extension)
    if not out_path.with_suffix(".mp4").exists():
        # search for any mp4 in out_dir matching the stem
        matches = list(Path(out_dir).glob(f"{out_path.stem}*.mp4"))
        if matches:
            matches[0].rename(out_path.with_suffix(".mp4"))
            return True
        return False
    return True


# ---------------------------------------------------------------------------
# Clip extraction + on-the-go cleanup
# ---------------------------------------------------------------------------

def get_duration(path: Path) -> float | None:
    """Get video duration in seconds via ffprobe (or ffmpeg fallback)."""
    info = _ffprobe_json(path)
    try:
        return float(info["format"].get("duration", ""))
    except (ValueError, KeyError):
        return None


def get_video_resolution(path: Path) -> tuple[int, int] | None:
    """Get video (width, height) via ffprobe (or ffmpeg fallback)."""
    info = _ffprobe_json(path)
    for s in info.get("streams", []):
        if s.get("codec_type") == "video" or "width" in s:
            w, h = s.get("width"), s.get("height")
            if w and h:
                return int(w), int(h)
    return None


def is_hd(path: Path) -> bool:
    """Check if a video file is HD (≥ 1280x720)."""
    res = get_video_resolution(path)
    if res is None:
        return False
    w, h = res
    return w >= HD_MIN_WIDTH and h >= HD_MIN_HEIGHT


def extract_clips(src: Path, out_dir: Path, title_id: str,
                  clip_min: float = CLIP_MIN_S, clip_max: float = CLIP_MAX_S,
                  step: float = CLIP_STEP_S) -> list[Path]:
    """Extract 4–15s clips from a source video using ffmpeg.

    Extracts one clip every `step` seconds, each `clip_max` seconds long
    (or shorter if near the end). Skips clips shorter than `clip_min`.

    Returns list of saved clip paths.
    """
    duration = get_duration(src)
    if duration is None or duration < clip_min:
        print(f"    skip (duration {duration}s < {clip_min}s)", flush=True)
        return []

    W, H = CANONICAL_RES
    clips = []
    start = 0.0
    idx = 0
    while start < duration:
        end = min(start + clip_max, duration)
        clip_dur = end - start
        if clip_dur < clip_min:
            break
        clip_path = out_dir / f"{title_id}_clip{idx:04d}.mp4"
        ff = _find_ffmpeg()
        cmd = [
            ff, "-y", "-ss", f"{start:.2f}", "-to", f"{end:.2f}",
            "-i", str(src),
            "-vf", f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
                   f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,fps={CANONICAL_FPS}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-ar", str(CANONICAL_AUDIO_SR), "-ac", str(CANONICAL_AUDIO_CH),
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            str(clip_path),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode == 0 and clip_path.exists():
            clips.append(clip_path)
            print(f"    clip {idx:04d}: {start:.1f}s–{end:.1f}s ({clip_dur:.1f}s) ✓",
                  flush=True)
        else:
            print(f"    clip {idx:04d}: FAIL {r.stderr[:150]}", flush=True)
        start += step
        idx += 1
    return clips


def dir_size_gb(path: Path) -> float:
    """Total size of a directory in GB."""
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total / (1024 ** 3)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    api_key_file: Path,
    out_dir: Path,
    max_size_gb: float = 500.0,
    max_titles: int = 500,
    resolution: str = "720",
    media_types: tuple[str, ...] = ("movie", "tv"),
    delete_source: bool = True,
    downloader_path: str | None = None,
    min_rating: float = 8.0,
) -> None:
    """Full TMDB → download → clip → cleanup pipeline.

    Args:
        api_key_file: path to the TMDB API key file.
        out_dir: output directory for clips.
        max_size_gb: stop when total clip storage exceeds this (default 500 GB).
        max_titles: max number of titles to process.
        resolution: download resolution ("720", "1080", "best").
        media_types: which media types to fetch ("movie", "tv").
        delete_source: if True, delete full source videos after clip extraction.
        downloader_path: path to vid-dl-cli or yt-dlp. Auto-detected if None.
        min_rating: only download titles with TMDB vote_average ≥ this (default 8.0).
    """
    token = load_api_key(api_key_file)
    print(f"TMDB API token loaded from {api_key_file}")

    downloader = downloader_path or find_vid_dl_cli()
    if not downloader:
        print("ERROR: no downloader found. Install vid-dl-cli-only or yt-dlp.",
              file=sys.stderr)
        sys.exit(1)
    print(f"Download backend: {downloader}")

    # Verify ffmpeg is available (via PATH or imageio_ffmpeg)
    try:
        ff = _find_ffmpeg()
        print(f"ffmpeg: {ff}")
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    clips_dir = out_dir / "clips"
    raw_dir = out_dir / "raw"
    clips_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.csv"
    manifest_fields = ["title_id", "title", "media_type", "tmdb_id", "year",
                       "genre", "n_clips", "clip_paths", "source_deleted",
                       "downloaded_at"]
    writer = csv.DictWriter(manifest_path.open("w", newline=""), fieldnames=manifest_fields)
    writer.writeheader()

    # --- Fetch title list from TMDB ---
    # Use top_rated as primary source (highest ratings), then trending as fallback.
    # All titles are filtered by min_rating (vote_average ≥ min_rating).
    titles: list[dict] = []
    for media_type in media_types:
        for source_name, fetch_fn in [
            ("top_rated", fetch_top_rated),
            ("trending", fetch_trending),
            ("popular", fetch_popular),
        ]:
            if len(titles) >= max_titles * 3:  # over-fetch so rating filter has enough
                break
            print(f"\nFetching {source_name} {media_type} from TMDB (min rating: {min_rating})...")
            for page in range(1, 11):  # up to 10 pages = 200 titles per source
                try:
                    if source_name == "trending":
                        data = fetch_fn(token, media_type, "week", page)
                    else:
                        data = fetch_fn(token, media_type, page)
                    for item in data.get("results", []):
                        vote = item.get("vote_average", 0)
                        if vote < min_rating:
                            continue  # skip low-rated titles
                        titles.append({
                            "media_type": media_type,
                            "tmdb_id": item["id"],
                            "title": item.get("title") or item.get("name", ""),
                            "year": (item.get("release_date") or item.get("first_air_date", ""))[:4],
                            "genre_ids": item.get("genre_ids", []),
                            "rating": vote,
                        })
                except Exception as e:
                    print(f"  page {page}: {e}")
                    break
                if len(titles) >= max_titles * 3:
                    break

    # Dedupe by (media_type, tmdb_id) and cap at max_titles
    seen_ids = set()
    deduped = []
    for t in titles:
        key = (t["media_type"], t["tmdb_id"])
        if key not in seen_ids:
            seen_ids.add(key)
            deduped.append(t)
    titles = deduped[:max_titles]

    print(f"\nTotal titles to process: {len(titles)} (rating ≥ {min_rating}, cap: {max_titles})")
    if titles:
        avg_rating = sum(t["rating"] for t in titles) / len(titles)
        print(f"Average rating: {avg_rating:.1f}/10")
    print(f"Storage cap: {max_size_gb} GB\n")

    # --- Process each title ---
    processed = 0
    for title in titles:
        # Check size cap
        current_size = dir_size_gb(clips_dir)
        if current_size >= max_size_gb:
            print(f"\n✋ Storage cap reached: {current_size:.1f} GB ≥ {max_size_gb} GB")
            print(f"   Stopping. {processed} titles processed, "
                  f"{len(list(clips_dir.glob('*.mp4')))} clips extracted.")
            break

        title_id = f"{title['media_type']}_{title['tmdb_id']}"
        print(f"\n[{processed+1}/{len(titles)}] {title['title']} ({title['year']}) "
              f"[{title['media_type']}] ⭐{title.get('rating', '?')}/10 "
              f"— storage: {current_size:.1f}/{max_size_gb} GB")

        # Build stream URL
        if title["media_type"] == "movie":
            stream_url = VIDSRC_MOVIE_URL.format(id=title["tmdb_id"])
        else:
            # For TV, download season 1 episode 1 as a representative
            stream_url = VIDSRC_TV_URL.format(id=title["tmdb_id"], season=1, episode=1)

        # Download
        raw_path = raw_dir / f"{title_id}"
        ok = download_video(stream_url, raw_path, downloader, resolution)
        if not ok:
            print(f"  download failed, skipping.")
            writer.writerow({
                "title_id": title_id, "title": title["title"],
                "media_type": title["media_type"], "tmdb_id": title["tmdb_id"],
                "year": title["year"], "genre": "", "n_clips": 0,
                "clip_paths": "", "source_deleted": False, "downloaded_at": "",
            })
            continue

        raw_file = raw_path.with_suffix(".mp4")
        if not raw_file.exists():
            print(f"  downloaded file not found, skipping.")
            continue

        # --- HD-only check: reject anything below 720p ---
        if not is_hd(raw_file):
            res = get_video_resolution(raw_file)
            print(f"  ✗ not HD (got {res}, need ≥{HD_MIN_WIDTH}x{HD_MIN_HEIGHT}), deleting & skipping")
            raw_file.unlink()
            writer.writerow({
                "title_id": title_id, "title": title["title"],
                "media_type": title["media_type"], "tmdb_id": title["tmdb_id"],
                "year": title["year"], "genre": "", "n_clips": 0,
                "clip_paths": "", "source_deleted": True,
                "downloaded_at": time.strftime("%Y-%m-%d %H:%M"),
            })
            continue
        print(f"  ✓ HD verified: {get_video_resolution(raw_file)}")

        # Extract clips
        clips = extract_clips(raw_file, clips_dir, title_id)
        print(f"  extracted {len(clips)} clips")

        # Delete source video on the go
        source_deleted = False
        if delete_source:
            raw_file.unlink()
            source_deleted = True
            print(f"  deleted source: {raw_file.name}")

        # Record in manifest
        writer.writerow({
            "title_id": title_id, "title": title["title"],
            "media_type": title["media_type"], "tmdb_id": title["tmdb_id"],
            "year": title["year"], "genre": ";".join(map(str, title["genre_ids"])),
            "n_clips": len(clips),
            "clip_paths": ";".join(c.name for c in clips),
            "source_deleted": source_deleted,
            "downloaded_at": time.strftime("%Y-%m-%d %H:%M"),
        })
        manifest_path.open("a").close()  # flush
        processed += 1

    final_size = dir_size_gb(clips_dir)
    total_clips = len(list(clips_dir.glob("*.mp4")))
    print(f"\n{'='*60}")
    print(f"Pipeline complete.")
    print(f"  Titles processed: {processed}")
    print(f"  Clips extracted:  {total_clips}")
    print(f"  Total storage:    {final_size:.1f} GB (cap: {max_size_gb})")
    print(f"  Clips directory:  {clips_dir}")
    print(f"  Manifest:         {manifest_path}")
    if delete_source:
        remaining = list(raw_dir.glob("*.mp4"))
        print(f"  Source videos remaining: {len(remaining)} (should be 0 if all succeeded)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="avforge.data.tmdb_pipeline",
        description="TMDB + Streambert/vid-dl → 4–15s clips pipeline for AV-Forge 2.0",
    )
    p.add_argument("--api-key-file", required=True,
                   help="path to TMDB API key file (e.g. D:\\AIModels\\TMDB_API.txt)")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--max-size-gb", type=float, default=500.0,
                   help="max total clip storage in GB (default: 500)")
    p.add_argument("--max-titles", type=int, default=500,
                   help="max number of titles to process (default: 500)")
    p.add_argument("--resolution", choices=list(HD_RESOLUTIONS), default="720",
                   help="download resolution — HD only (default: 720, min 720p)")
    p.add_argument("--media-types", nargs="+", default=["movie", "tv"],
                   choices=["movie", "tv"], help="media types to fetch")
    p.add_argument("--no-delete-source", action="store_true",
                   help="keep full source videos after clip extraction (default: delete)")
    p.add_argument("--downloader", default=None,
                   help="path to vid-dl-cli or yt-dlp executable (auto-detected if omitted)")
    p.add_argument("--min-rating", type=float, default=8.0,
                   help="only download titles with TMDB rating ≥ this (default: 8.0)")
    args = p.parse_args(argv)

    run_pipeline(
        api_key_file=Path(args.api_key_file),
        out_dir=Path(args.out),
        max_size_gb=args.max_size_gb,
        max_titles=args.max_titles,
        resolution=args.resolution,
        media_types=tuple(args.media_types),
        delete_source=not args.no_delete_source,
        downloader_path=args.downloader,
        min_rating=args.min_rating,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())