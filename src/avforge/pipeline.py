"""End-to-end generation pipeline for AV-Forge 2.0.

Ties together encoders, reference registry, MMDiT backbone, sampler, and VAE
decoders. Exposes a single `generate()` entry point matching the Seedance 2.0
report's input/output contract (§1.1–1.2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from .config import Config, InferenceConfig, Resolution, Variant
from .encoders import build_encoders
from .model import MMDiTBackbone
from .registry import ReferenceRegistry, build_registry
from .sampler import sample


@dataclass
class GenerationRequest:
    """A single generation request, matching the Seedance 2.0 input contract.

    All reference fields are optional and combinable (any combination is valid).
    """
    text: str = ""
    subject_images: Sequence[Tensor] | None = None
    subject_videos: Sequence[Tensor] | None = None
    motion_videos: Sequence[Tensor] | None = None
    style_images: Sequence[Tensor] | None = None
    style_videos: Sequence[Tensor] | None = None
    vfx_references: Sequence[Tensor] | None = None
    audio_references: Sequence[tuple[Tensor, str]] | None = None
    first_frame: Tensor | None = None
    last_frame: Tensor | None = None
    edit_instruction: Tensor | None = None
    source_clip: tuple[Tensor, Tensor] | None = None
    duration_s: float = 10.0
    resolution: Resolution = Resolution.P720
    variant: Variant = Variant.FULL
    seed: int | None = None


@dataclass
class GenerationResult:
    video: Tensor          # (B, C, T, H, W) decoded frames
    audio: Tensor          # (B, 2, samples) binaural waveform
    shot_plan: Tensor       # autonomous shot plan from AR prefix
    registry_desc: str      # human-readable reference/task description


class AVForgePipeline:
    """Full generation pipeline. In a real deployment this loads pretrained weights;
    here it builds the architecture and runs the (untrained) forward pass so the
    scaffold is executable and shape-correct.
    """

    def __init__(self, cfg: Config = Config()):
        self.cfg = cfg
        self.encoders = build_encoders(cfg.model)
        self.backbone = MMDiTBackbone(cfg.model)
        self._device = None

    def _to(self, device, dtype):
        for m in self.encoders.values():
            m.to(device=device, dtype=dtype)
        self.backbone.to(device=device, dtype=dtype)
        self._device = device

    @torch.no_grad()
    def generate(self, req: GenerationRequest) -> GenerationResult:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32
        self._to(device, dtype)

        # --- Encode text ---
        # In a real impl, tokenize req.text with the text encoder's tokenizer.
        N_text = max(1, len(req.text.split()))
        text_ids = torch.zeros(1, N_text, dtype=torch.long, device=device)
        text_tokens, _ = self.encoders["text"](text_ids)

        # --- Build reference registry ---
        registry = build_registry(
            subject_images=req.subject_images,
            subject_videos=req.subject_videos,
            motion_videos=req.motion_videos,
            style_images=req.style_images,
            style_videos=req.style_videos,
            vfx_references=req.vfx_references,
            audio_references=req.audio_references,
            first_frame=req.first_frame,
            last_frame=req.last_frame,
            edit_instruction=req.edit_instruction,
            source_clip=req.source_clip,
        )

        # --- Compute latent shape ---
        # For the smoke-test scaffold we cap the token grid so full-attention is
        # tractable on small GPUs. A real deployment uses windowed attention
        # (see docs/02_ARCHITECTURE.md §1.4) and the full grid below.
        T_frames = int(req.duration_s * self.cfg.io.fps)
        H = 270 if req.resolution == Resolution.P480 else 540
        W = 480 if req.resolution == Resolution.P480 else 960
        Cv = self.cfg.model.video_vae_channels
        Ca = self.cfg.model.audio_vae_channels
        # Tiny smoke-test grid: 4 temporal x 8x8 spatial video tokens + 16 audio
        is_tiny = self.cfg.model.core_hidden <= 512
        if is_tiny:
            Nv = 4 * 8 * 8      # 256 video tokens
            Na = 16             # 16 audio tokens
        else:
            Nv = (T_frames // self.cfg.model.video_vae_temporal_downsample) * (H // 8) * (W // 8)
            Na = int(req.duration_s * self.cfg.model.audio_frame_rate_hz) // 4
        D = self.cfg.model.core_hidden
        shape = (1, Nv + Na, Cv + Ca)

        # --- AR shot-plan tokens (placeholder) ---
        shot_plan_tokens = torch.zeros(1, 8, dtype=torch.long, device=device)

        # --- Sample ---
        inf = InferenceConfig(variant=req.variant, seed=req.seed)
        v_latent, a_latent = sample(
            self.backbone, shape, text_tokens, registry, inf, device, dtype,
            shot_plan_tokens, Nv, Na,
        )

        # --- Decode ---
        # Reshape video latent to (B, C, T, H, W) for the VAE decoder
        if is_tiny:
            T_lat, H_lat, W_lat = 4, 8, 8
        else:
            T_lat = T_frames // self.cfg.model.video_vae_temporal_downsample
            H_lat, W_lat = H // 8, W // 8
        v_latent_5d = v_latent[:, :T_lat * H_lat * W_lat].reshape(1, Cv, T_lat, H_lat, W_lat)
        video = self.encoders["video_vae"].decode(v_latent_5d)
        # Audio latent -> waveform. conv1d expects (B, C, L).
        a_latent_3d = a_latent.transpose(1, 2).contiguous()  # (1, Ca, Na)
        audio = self.encoders["audio_vae"].decode(a_latent_3d)

        return GenerationResult(
            video=video,
            audio=audio,
            shot_plan=torch.zeros(1, 8, 7),  # placeholder
            registry_desc=registry.describe(),
        )