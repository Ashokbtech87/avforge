"""Data collection toolkit for AV-Forge 2.0.

Implements the operational pipeline from docs/06_DATA_COLLECTION.md:
  - normalize: decode + re-encode raw clips to canonical format (ffmpeg)
  - balance: per-category balancing + dedupe
  - shard: write WebDataset-style .tar shards for training
  - source downloaders: stubs for AudioSet / Common Voice / Freesound / YouTube

Usage:
  python -m avforge.data.collect --normalize data/raw --out data/normalized --fps 24 --res 720p
  python -m avforge.data.collect --balance data/annotated --out data/shards --shard-size 1000
  python -m avforge.data.collect --source audioset --out data/raw/audioset --limit 1000
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Canonical format (matches docs/02_ARCHITECTURE.md §1.2)
CANONICAL_FPS = 24
CANONICAL_RES = {"480p": (854, 480), "720p": (1280, 720)}
CANONICAL_AUDIO_SR = 24000
CANONICAL_AUDIO_CH = 2
CANONICAL_DURATION_RANGE = (4.0, 15.0)


def _run(cmd: list[str]) -> tuple[int, str]:
    """Run a command, return (exitcode, output)."""
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def get_clip_duration(src: Path) -> float | None:
    """Get clip duration in seconds via ffprobe. Returns None on failure."""
    r = _run(["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
              "-of", "default=noprint_wrappers=1:nokey=1", str(src)])
    try:
        return float(r[1].strip()) if r[0] == 0 else None
    except (ValueError, IndexError):
        return None


def normalize_clip(src: Path, dst: Path, fps: int = CANONICAL_FPS,
                   res: str = "720p", sr: int = CANONICAL_AUDIO_SR) -> bool:
    """Re-encode a clip to canonical format using ffmpeg.

    Skips clips shorter than 4s (too short for the model's 4–15s range).
    Trims clips longer than 15s to 15s. Sets fps, resolution, audio rate + channels.
    Returns True on success.
    """
    # Enforce minimum duration — skip clips shorter than 4s
    dur = get_clip_duration(src)
    if dur is not None and dur < CANONICAL_DURATION_RANGE[0]:
        print(f" SKIP (too short: {dur:.1f}s < {CANONICAL_DURATION_RANGE[0]}s)", end="")
        return False

    W, H = CANONICAL_RES[res]
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-t", str(CANONICAL_DURATION_RANGE[1]),   # cap at 15s
        "-vf", f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,fps={fps}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-ar", str(sr), "-ac", str(CANONICAL_AUDIO_CH),
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(dst),
    ]
    code, out = _run(cmd)
    if code != 0:
        print(f"  FAIL {src.name}: {out[:200]}", file=sys.stderr)
        return False
    return True


def phash_file(path: Path, n: int = 65536) -> str:
    """Cheap perceptual-ish hash (first N bytes + size) for dedupe.
    A real impl uses imageio + imagehash on a middle frame."""
    h = hashlib.sha256()
    h.update(str(path.stat().st_size).encode())
    with open(path, "rb") as f:
        h.update(f.read(n))
    return h.hexdigest()


def normalize_dir(in_dir: Path, out_dir: Path, fps: int, res: str) -> None:
    """Normalize all video/audio clips in a directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    exts = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".wav", ".mp3", ".m4a", ".flac"}
    files = [f for f in in_dir.rglob("*") if f.suffix.lower() in exts]
    print(f"Normalizing {len(files)} clips from {in_dir} → {out_dir} ({res}, {fps}fps)")
    ok = 0
    for i, src in enumerate(files):
        dst = out_dir / (src.stem + ".mp4")
        if dst.exists():
            continue
        print(f"  [{i+1}/{len(files)}] {src.name}", end="", flush=True)
        if normalize_clip(src, dst, fps, res):
            ok += 1
            print(" ✓")
        else:
            print(" ✗")
    print(f"Done: {ok}/{len(files)} normalized.")


def balance_and_shard(annot_dir: Path, out_dir: Path, shard_size: int) -> None:
    """Read annotated manifest, balance per category, dedupe, write shards.

    Expects an `manifest.csv` in annot_dir with columns:
      id, path, categories (semicolon-separated), phash, duration, license
    Writes WebDataset-style .tar shards to out_dir, each with <= shard_size clips.
    """
    manifest = annot_dir / "manifest.csv"
    if not manifest.exists():
        print(f"ERROR: {manifest} not found. Run annotate first.", file=sys.stderr)
        sys.exit(1)

    rows = list(csv.DictReader(manifest.open()))
    print(f"Loaded {len(rows)} annotated clips.")

    # Dedupe by phash
    seen = set()
    deduped = []
    for r in rows:
        h = r.get("phash", "")
        if h and h in seen:
            continue
        seen.add(h)
        deduped.append(r)
    print(f"After dedupe: {len(deduped)} clips.")

    # Balance: group by category, cap per category (simple even split)
    from collections import defaultdict
    by_cat = defaultdict(list)
    for r in deduped:
        for c in r.get("categories", "").split(";"):
            c = c.strip()
            if c:
                by_cat[c].append(r)

    # Even cap: total target / num categories
    total_target = len(deduped)
    cap = max(100, total_target // max(1, len(by_cat)))
    balanced = []
    for cat, items in by_cat.items():
        balanced.extend(items[:cap])
    print(f"After balancing across {len(by_cat)} categories: {len(balanced)} clips.")

    # Shard into tars
    out_dir.mkdir(parents=True, exist_ok=True)
    n_shards = (len(balanced) + shard_size - 1) // shard_size
    for si in range(n_shards):
        shard = balanced[si * shard_size: (si + 1) * shard_size]
        shard_path = out_dir / f"shard-{si:05d}.tar"
        # Write a simple tar (real impl uses webdataset/tarwriter)
        import tarfile
        with tarfile.open(shard_path, "w") as tar:
            for r in shard:
                p = Path(r["path"])
                if p.exists():
                    tar.add(p, arcname=p.name)
                # add metadata json
                info = tarfile.TarInfo(p.stem + ".json")
                data = json.dumps(r).encode()
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        print(f"  wrote {shard_path} ({len(shard)} clips)")
    print(f"Done: {n_shards} shards in {out_dir}.")


# --- Source downloaders (stubs — fill in with your API keys) ---

def download_audioset(out_dir: Path, limit: int) -> None:
    """Download AudioSet clips. Requires the AudioSet CSV + GCS access.
    Stub: see https://research.google.com/audioset/download.html"""
    out_dir.mkdir(parents=True, exist_ok=True)
    print("AudioSet downloader (stub):")
    print("  1. Download the balanced_train_segments.csv from AudioSet.")
    print("  2. For each row, fetch the 10s clip from GCS:")
    print("     gs://audioset/youtube_corpus_v1/...")
    print(f"  Target: {limit} clips → {out_dir}")
    print("  Fill this in with `gsutil cp` or the AudioSet API.")


def download_common_voice(out_dir: Path, limit: int) -> None:
    """Mozilla Common Voice — multilingual speech."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Common Voice downloader (stub):")
    print("  1. Visit https://commonvoice.mozilla.org/en/datasets")
    print("  2. Download language packs (en, ja, ko, id, pt, es + dialects).")
    print(f"  Target: {limit} clips → {out_dir}")


def download_freesound(out_dir: Path, limit: int, api_key: str | None) -> None:
    """Freesound SFX via API. Requires FREESOUND_API_KEY env."""
    out_dir.mkdir(parents=True, exist_ok=True)
    key = api_key or os.environ.get("FREESOUND_API_KEY")
    if not key:
        print("Freesound: set FREESOUND_API_KEY env var to download.")
        return
    print(f"Freesound downloader (stub with key): would fetch {limit} SFX → {out_dir}")
    # Real impl: requests.get(f"https://freesound.org/apiv2/search/text/?query=...&token={key}")


def download_youtube(out_dir: Path, limit: int, playlist: str | None) -> None:
    """YouTube via yt-dlp (with creator permission / CC-BY only)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if shutil.which("yt-dlp") is None:
        print("yt-dlp not installed. Run: pip install yt-dlp")
        return
    if not playlist:
        print("YouTube: provide --playlist URL (CC-BY or with permission).")
        return
    cmd = ["yt-dlp", "-f", "mp4", "--max-downloads", str(limit),
           "-o", str(out_dir / "%(id)s.%(ext)s"), playlist]
    print(f"Downloading up to {limit} from {playlist}")
    _run(cmd)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="avforge.data.collect",
                               description="AV-Forge 2.0 data collection toolkit")
    sub = p.add_subparsers(dest="cmd", required=True)

    n = sub.add_parser("normalize", help="Re-encode clips to canonical format")
    n.add_argument("in_dir")
    n.add_argument("--out", required=True)
    n.add_argument("--fps", type=int, default=CANONICAL_FPS)
    n.add_argument("--res", choices=list(CANONICAL_RES), default="720p")
    n.set_defaults(func=lambda a: normalize_dir(Path(a.in_dir), Path(a.out), a.fps, a.res))

    b = sub.add_parser("balance", help="Balance + dedupe + shard annotated data")
    b.add_argument("in_dir")
    b.add_argument("--out", required=True)
    b.add_argument("--shard-size", type=int, default=1000)
    b.set_defaults(func=lambda a: balance_and_shard(Path(a.in_dir), Path(a.out), a.shard_size))

    d = sub.add_parser("download", help="Download from a public source")
    d.add_argument("--source", choices=["audioset", "commonvoice", "freesound", "youtube"], required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--limit", type=int, default=1000)
    d.add_argument("--playlist", default=None, help="YouTube playlist URL")
    d.add_argument("--api-key", default=None)
    d.set_defaults(func=lambda a: {
        "audioset": download_audioset, "commonvoice": download_common_voice,
        "freesound": download_freesound, "youtube": download_youtube,
    }[a.source](Path(a.out), a.limit, a.api_key or a.playlist))

    args = p.parse_args(argv)
    if callable(getattr(args, "func", None)):
        args.func(args)
        return 0
    p.print_help()
    return 1


import io  # used in balance_and_shard

if __name__ == "__main__":
    raise SystemExit(main())