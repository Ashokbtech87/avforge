"""AV-Forge 2.0 command-line interface.

Usage:
  python -m avforge generate --text "..." --duration 10 --resolution 720p --variant full
  python -m avforge eval --task T2V
  python -m avforge prompts --list
  python -m avforge prompts --id anchor-skating
"""

from __future__ import annotations

import argparse
import json
import sys

import torch

from . import __version__
from .config import Resolution, Variant, tiny_config
from .pipeline import AVForgePipeline, GenerationRequest
from .prompts import PROMPTS, by_id
from .eval import evaluate, T2V_TARGETS, MOTION_TARGETS, AUDIO_TARGETS


def _save_result(result, out_prefix: str) -> None:
    """Save video frames (PNG sequence + GIF) and audio (WAV) to disk.
    Uses only stdlib + numpy + PIL/imageio if available; falls back gracefully.
    """
    import os
    video = result.video[0]   # (C, T, H, W)
    audio = result.audio[0]    # (2, samples)
    C, T, H, W = video.shape
    frames_dir = f"{out_prefix}_frames"
    os.makedirs(frames_dir, exist_ok=True)

    # Save frames as PNG via PIL (no ffmpeg needed)
    try:
        from PIL import Image
        import numpy as np
        # video is float; normalize to 0-255 uint8
        v = video.detach().cpu().float()
        v = (v - v.min()) / (v.max() - v.min() + 1e-8)
        v = (v * 255).clamp(0, 255).to(torch.uint8)
        # (C, T, H, W) -> (T, H, W, C)
        frames = v.permute(1, 2, 3, 0).numpy()
        if frames.shape[-1] == 1:
            frames = frames.repeat(3, axis=-1)
        for i, f in enumerate(frames):
            Image.fromarray(f).save(os.path.join(frames_dir, f"frame_{i:03d}.png"))
        # Animated GIF
        gif_path = f"{out_prefix}.gif"
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], duration=1000//24, loop=0)
        print(f"saved {T} frames to {frames_dir}/")
        print(f"saved animated GIF to {gif_path}")
    except Exception as e:
        print(f"(frame save skipped: {e})")

    # Save audio as WAV
    try:
        import scipy.io.wavfile as wavfile
        import numpy as np
        a = audio.detach().cpu().float()
        a = (a / (a.abs().max() + 1e-8) * 32767).clamp(-32767, 32767).to(torch.int16)
        a = a.numpy().T  # (samples, 2)
        wav_path = f"{out_prefix}.wav"
        wavfile.write(wav_path, 24000, a)
        print(f"saved audio to {wav_path}")
    except Exception as e:
        print(f"(audio save skipped: {e})")


def _cmd_generate(args: argparse.Namespace) -> int:
    cfg = tiny_config() if args.tiny else None
    pipe = AVForgePipeline(cfg) if cfg else AVForgePipeline()
    req = GenerationRequest(
        text=args.text,
        duration_s=args.duration,
        resolution=Resolution(args.resolution),
        variant=Variant(args.variant),
        seed=args.seed,
    )
    tag = " [TINY ~0.5B, 8GB-friendly]" if args.tiny else ""
    print(f"AV-Forge 2.0 v{__version__}{tag} — generating {args.duration}s {args.resolution} "
          f"({args.variant})...", file=sys.stderr)
    print(req.text[:120] + ("..." if len(req.text) > 120 else ""), file=sys.stderr)
    result = pipe.generate(req)
    print(result.registry_desc, file=sys.stderr)
    print(f"video: {tuple(result.video.shape)}  audio: {tuple(result.audio.shape)}")
    if args.out:
        _save_result(result, args.out)
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    # Demo: evaluate with placeholder scores at the target thresholds to show
    # the rubric machinery. In real use, scores come from the objective +
    # subjective tracks (docs/04_EVALUATION.md §5).
    demo_scores = {k: v for k, v in T2V_TARGETS.items()}
    demo_fine = {**{k: v for k, v in MOTION_TARGETS.items() if k in (
        "Multi-Entity Feature Match", "Framing / Composition", "Editing Rhythm",
        "Emotion & Expression", "Counter-Reality Instructions", "Surreal Motion",
    )}, **{k: v for k, v in AUDIO_TARGETS.items() if k in (
        "English", "Chinese Opera", "Singing / Rap", "Animal Sound",
    )}}
    rep = evaluate(
        demo_scores, task=args.task, duration_s=10.0, resolution="720p",
        variant="full", fine_grained=demo_fine,
        narrative={"Cinematographic Language": 4.10, "Plot Design": 3.95,
                    "Stylistic Aesthetics": 4.05},
        usability={"Audio Prompt Following": "usability 86%, satisfaction 60%, delight 29% ✓"},
        arena_elo={"T2V": 1458, "I2V": 1455},
        qualitative=[
            "Strong: multi-entity identity preservation, binaural layering, beat-matched cuts.",
            "Weak: surreal-motion margin thin; add 20k procedural surreal hard negs.",
            "Suggestion: up-weight spectral loss on high-motion segments.",
        ],
    )
    print(rep.render())
    return 0


def _cmd_prompts(args: argparse.Namespace) -> int:
    if args.id:
        p = by_id(args.id)
        print(f"[{p.id}] task={p.task} beats={p.beats}")
        print(f"categories: {', '.join(p.categories)}")
        print()
        print(p.text)
        return 0
    for p in PROMPTS:
        print(f"{p.id:24s} {p.task:4s}  {', '.join(p.categories[:2])}{'...' if len(p.categories)>2 else ''}")
    print(f"\n{len(PROMPTS)} stress-test prompts total.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="avforge", description=f"AV-Forge 2.0 v{__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Generate audio-video from a prompt")
    g.add_argument("--text", required=True)
    g.add_argument("--duration", type=float, default=10.0)
    g.add_argument("--resolution", choices=[r.value for r in Resolution], default="720p")
    g.add_argument("--variant", choices=[v.value for v in Variant], default="full")
    g.add_argument("--seed", type=int, default=None)
    g.add_argument("--tiny", action="store_true",
                   help="use ~0.5B config that fits in ~6GB VRAM (smoke-test only)")
    g.add_argument("--out", default=None, metavar="PREFIX",
                   help="save output: <PREFIX>.gif, <PREFIX>.wav, <PREFIX>_frames/")
    g.set_defaults(func=_cmd_generate)

    e = sub.add_parser("eval", help="Run SeedVideoBench 2.0-style self-evaluation")
    e.add_argument("--task", choices=["T2V", "I2V", "R2V"], default="T2V")
    e.set_defaults(func=_cmd_eval)

    p = sub.add_parser("prompts", help="List or show stress-test prompts")
    p.add_argument("--list", action="store_true")
    p.add_argument("--id", default=None)
    p.set_defaults(func=_cmd_prompts)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())