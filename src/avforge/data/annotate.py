"""Auto-annotation pipeline for AV-Forge 2.0 training data.

Implements the annotation steps from docs/06_DATA_COLLECTION.md §2.4:
  - shot detection + camera grammar
  - OCR (text regions)
  - forced alignment (audio)
  - speaker/entity IDs
  - palette tokens
  - physics tags
  - aesthetics scores
  - category tagging (maps to 03_DATA.md fine-grained categories)

Usage:
  python -m avforge.data.annotate --in data/normalized --out data/annotated

NOTE: This is a scaffold. The heavy annotators (Whisper, EasyOCR, InsightFace,
pyannote, PySceneDetect) are imported lazily and require their own deps.
Install them with: pip install whisper easyocr insightface pyannote.audio scenedetect
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from ..eval import MOTION_CATEGORIES, AUDIO_CATEGORIES


def _safe_import(mod: str):
    try:
        return __import__(mod)
    except ImportError:
        print(f"  (optional dep '{mod}' not installed — skipping that annotator)", file=sys.stderr)
        return None


def detect_shots(video_path: Path) -> list[dict]:
    """Shot boundaries + camera-move classification via PySceneDetect."""
    scenedetect = _safe_import("scenedetect")
    if scenedetect is None:
        return [{"start": 0, "end": 1, "move": "unknown"}]
    # Real impl: SceneManager + ContentDetector + a camera-move classifier.
    return [{"start": 0, "end": 1, "move": "static"}]


def ocr_frames(video_path: Path) -> list[dict]:
    """Text regions via EasyOCR. Returns per-frame text + bounding boxes."""
    easyocr = _safe_import("easyocr")
    if easyocr is None:
        return []
    # Real impl: reader = easyocr.Reader(['ch_sim','en']); read middle frames.
    return []


def forced_align(video_path: Path) -> dict:
    """Per-phoneme timestamps via Whisper + Montreal Forced Aligner."""
    whisper = _safe_import("whisper")
    if whisper is None:
        return {"transcript": "", "phonemes": []}
    # Real impl: model = whisper.load_model("base"); result = model.transcribe(video_path)
    return {"transcript": "", "phonemes": []}


def cluster_entities(video_path: Path) -> dict:
    """Face + voice clustering → stable entity/speaker IDs."""
    insightface = _safe_import("insightface")
    pyannote = _safe_import("pyannote.audio")
    return {"face_ids": [], "voice_ids": []}


def palette_token(video_path: Path) -> list[float]:
    """16-dim color-histogram palette embedding."""
    import random
    # Real impl: extract middle frame, compute histogram in HSV, reduce to 16 dims.
    return [round(random.random(), 4) for _ in range(16)]


def physics_tags(video_path: Path) -> dict:
    """Contact/support/momentum tags (lightweight estimator)."""
    return {"contact": False, "support": False, "momentum": False}


def aesthetics_score(video_path: Path) -> float:
    """1–5 aesthetics score from a human-preference model."""
    import random
    # Real impl: load trained preference model, score middle frame + motion.
    return round(random.uniform(2.5, 4.5), 2)


def tag_categories(meta: dict) -> list[str]:
    """Map annotation metadata to fine-grained categories from 03_DATA.md."""
    cats = []
    # Heuristic mapping based on available signals.
    if meta["shots"][0]["move"] not in ("static", "unknown"):
        cats.append("Advanced Camera Movement")
    if meta["ocr"]:
        cats.extend(["Text Overlay", "Short Text", "Creative Text"])
    if meta["alignment"]["transcript"]:
        cats.append("English")  # language detection would refine this
    if meta["entities"]["voice_ids"]:
        cats.append("Voice + Action Interaction")
    if not cats:
        cats.append("Basic Scene")
    # Dedupe + intersect with known categories
    known = set(MOTION_CATEGORIES) | set(AUDIO_CATEGORIES)
    return [c for c in dict.fromkeys(cats) if c in known or c == "Basic Scene"]


def annotate_clip(video_path: Path) -> dict:
    """Run all annotators on one clip, return a metadata dict."""
    return {
        "id": video_path.stem,
        "path": str(video_path),
        "shots": detect_shots(video_path),
        "ocr": ocr_frames(video_path),
        "alignment": forced_align(video_path),
        "entities": cluster_entities(video_path),
        "palette": palette_token(video_path),
        "physics": physics_tags(video_path),
        "aesthetics": aesthetics_score(video_path),
    }


def annotate_dir(in_dir: Path, out_dir: Path) -> None:
    """Annotate all clips in a directory, write manifest.csv + per-clip JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)
    exts = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
    files = [f for f in in_dir.rglob("*") if f.suffix.lower() in exts]
    print(f"Annotating {len(files)} clips from {in_dir} → {out_dir}")

    manifest_path = out_dir / "manifest.csv"
    fields = ["id", "path", "categories", "phash", "duration", "license",
              "aesthetics", "n_shots", "n_entities", "has_text", "transcript"]
    writer = csv.DictWriter(manifest_path.open("w", newline=""), fieldnames=fields)
    writer.writeheader()

    for i, src in enumerate(files):
        print(f"  [{i+1}/{len(files)}] {src.name}", flush=True)
        meta = annotate_clip(src)
        meta["categories"] = tag_categories(meta)
        # write per-clip JSON
        (out_dir / f"{src.stem}.json").write_text(json.dumps(meta, indent=2))
        # write manifest row
        writer.writerow({
            "id": meta["id"],
            "path": meta["path"],
            "categories": ";".join(meta["categories"]),
            "phash": "",  # filled by collect.py balance step
            "duration": "",
            "license": "",
            "aesthetics": meta["aesthetics"],
            "n_shots": len(meta["shots"]),
            "n_entities": len(meta["entities"]["face_ids"]),
            "has_text": bool(meta["ocr"]),
            "transcript": meta["alignment"]["transcript"][:200],
        })
    print(f"Done. Manifest: {manifest_path}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="avforge.data.annotate")
    p.add_argument("--in", dest="in_dir", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    annotate_dir(Path(args.in_dir), Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())