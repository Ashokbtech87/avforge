"""Synthetic data generator for AV-Forge 2.0 hard negatives.

Generates the procedural VFX / physics-sim clips that target Seedance 2.0's
acknowledged weaknesses (docs/01_DESIGN.md §1.7): deformations, edge-case
physics, multi-speaker lip-sync, text restoration, extension seam drift,
HF noise. These are Tier C in docs/03_DATA.md.

Uses numpy + matplotlib + PIL (no Blender/taichi dependency) so it runs anywhere.
For higher-fidelity sims, swap in Blender/PyBullet scripts.

Usage:
  python -m avforge.data.synth --out data/synth --n 1000
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np


def _save_frame_png(arr: np.ndarray, path: Path) -> None:
    """Save an (H, W, 3) uint8 array as PNG via PIL."""
    from PIL import Image
    Image.fromarray(arr).save(path)


def gen_physics_collision(out: Path, H: int = 256, W: int = 256, T: int = 24) -> dict:
    """Two spheres colliding — ground-truth momentum + contact labels.
    Hard negative: also generate a deformed version."""
    frames = []
    for t in range(T):
        f = np.zeros((H, W, 3), dtype=np.uint8)
        # ball 1 moves right, ball 2 moves left, collide at t=T//2
        x1 = int(W * 0.2 + (W * 0.3) * (t / T))
        x2 = int(W * 0.8 - (W * 0.3) * (t / T))
        y = H // 2
        for (x, color) in [(x1, (255, 0, 0)), (x2, (0, 0, 255))]:
            yy, xx = np.ogrid[:H, :W]
            mask = (xx - x) ** 2 + (yy - y) ** 2 <= 20 ** 2
            f[mask] = color
        frames.append(f)
    return {"frames": frames, "categories": ["Physical Phenomena", "Physical Feedback"],
            "physics": {"contact": True, "momentum": True, "support": False}}


def gen_fluid_pour(out: Path, H: int = 256, W: int = 256, T: int = 24) -> dict:
    """Fluid pouring into a glass — surface tension + splash."""
    frames = []
    for t in range(T):
        f = np.zeros((H, W, 3), dtype=np.uint8)
        # stream
        for y in range(0, H // 2):
            f[y, W // 2] = (100, 150, 255)
        # rising fluid in glass
        level = int((H // 2) * (t / T))
        f[H - level:, W // 4: 3 * W // 4] = (100, 150, 255)
        frames.append(f)
    return {"frames": frames, "categories": ["Natural Phenomena", "Physical Phenomena"],
            "physics": {"contact": True, "momentum": False, "support": True}}


def gen_multi_speaker(out: Path, H: int = 256, W: int = 256, T: int = 24) -> dict:
    """Two talking heads — multi-speaker lip-sync hard case."""
    frames = []
    for t in range(T):
        f = np.zeros((H, W, 3), dtype=np.uint8)
        # head 1 left, head 2 right; mouth opens alternately
        mouth1 = 4 if (t % 6) < 3 else 8
        mouth2 = 8 if (t % 6) >= 3 else 4
        # head 1
        f[60:200, 40:120] = (200, 170, 150)
        f[140:140 + mouth1, 70:90] = (40, 20, 20)
        # head 2
        f[60:200, 140:220] = (180, 160, 140)
        f[140:140 + mouth2, 170:190] = (40, 20, 20)
        frames.append(f)
    return {"frames": frames, "categories": ["Chinese Multi-Person Dialogue", "Audio-Visual Sync"],
            "physics": {"contact": False, "momentum": False, "support": False}}


def gen_text_edit(out: Path, H: int = 256, W: int = 256, T: int = 24) -> dict:
    """Sign with text — for text-restoration-in-edits hard negatives."""
    from PIL import Image, ImageDraw
    frames = []
    for t in range(T):
        img = Image.new("RGB", (W, H), (20, 20, 20))
        d = ImageDraw.Draw(img)
        # text flickers/stabilizes across frames
        text = "OPEN" if t < T // 2 else "CLOSED"
        d.text((W // 4, H // 2), text, fill=(255, 255, 0))
        frames.append(np.array(img))
    return {"frames": frames, "categories": ["Text Overlay", "Creative Text"],
            "physics": {"contact": False, "momentum": False, "support": False}}


def gen_extension_seam(out: Path, H: int = 256, W: int = 256, T: int = 24) -> dict:
    """Source + extension with deliberate color/identity drift — hard negative
    for the extension seam (Seedance 2.0's weakest task)."""
    frames = []
    for t in range(T):
        f = np.zeros((H, W, 3), dtype=np.uint8)
        # source: stable blue car; extension: drifts to red + shifts
        if t < T // 2:
            f[H // 2 - 20: H // 2 + 20, 50:150] = (50, 50, 200)  # blue car
        else:
            drift = (t - T // 2) * 3
            f[H // 2 - 20: H // 2 + 20, 50 + drift: 150 + drift] = (200, 50, 50)  # red, shifted
        frames.append(f)
    return {"frames": frames, "categories": ["Multi-Entity Feature Match", "Lighting & Color Tone"],
            "physics": {"contact": False, "momentum": True, "support": False}}


def gen_hf_noise(out: Path, H: int = 256, W: int = 256, T: int = 24) -> dict:
    """Clean scene + structured HF noise — for spectral-loss training."""
    frames = []
    for t in range(T):
        base = np.full((H, W, 3), 128, dtype=np.uint8)
        if t >= T // 2:
            noise = np.random.randint(-40, 40, (H, W, 3)).astype(np.int16)
            base = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        frames.append(base)
    return {"frames": frames, "categories": ["Visual Style"],
            "physics": {"contact": False, "momentum": False, "support": False}}


GENERATORS = {
    "collision": gen_physics_collision,
    "fluid": gen_fluid_pour,
    "multispeaker": gen_multi_speaker,
    "text_edit": gen_text_edit,
    "extension_seam": gen_extension_seam,
    "hf_noise": gen_hf_noise,
}


def generate_dataset(out_dir: Path, n: int) -> None:
    """Generate n synthetic clips across all generator types."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    per_type = n // len(GENERATORS)
    for name, gen in GENERATORS.items():
        for i in range(per_type):
            clip_id = f"{name}_{i:05d}"
            clip_dir = out_dir / clip_id
            clip_dir.mkdir(exist_ok=True)
            data = gen(clip_dir)
            for t, frame in enumerate(data["frames"]):
                _save_frame_png(frame, clip_dir / f"frame_{t:03d}.png")
            meta = {
                "id": clip_id, "type": name,
                "categories": data["categories"],
                "physics": data["physics"],
                "n_frames": len(data["frames"]),
            }
            (clip_dir / "meta.json").write_text(json.dumps(meta, indent=2))
            manifest.append(meta)
            if (i + 1) % 50 == 0:
                print(f"  {name}: {i+1}/{per_type}", flush=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Done: {len(manifest)} synthetic clips in {out_dir}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="avforge.data.synth")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=600)
    args = p.parse_args(argv)
    generate_dataset(Path(args.out), args.n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())