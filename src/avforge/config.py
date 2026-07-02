"""Configuration for AV-Forge 2.0.

All sizes/resolutions/durations follow the Seedance 2.0 report spec:
  - Output: 4–15s, 480p or 720p, binaural audio.
  - Inputs: text + up to 9 images + up to 3 videos + up to 3 audio clips.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Resolution(str, Enum):
    P480 = "480p"
    P720 = "720p"


class Variant(str, Enum):
    FULL = "full"      # 50-step DPM-Solver + CFG
    FAST = "fast"      # 4–8 step consistency-distilled


@dataclass(frozen=True)
class ModelConfig:
    """Backbone sizing (reference design from docs/02_ARCHITECTURE.md)."""

    # MMDiT core
    core_layers: int = 48
    core_hidden: int = 2048
    core_heads: int = 32
    core_params_b: float = 7.0

    # AR shot-planner prefix
    ar_layers: int = 12
    ar_params_b: float = 1.0

    # Encoders (frozen at inference)
    text_encoder_params_b: float = 7.0
    image_vit_layers: int = 24
    video_encoder_layers: int = 24
    audio_encoder_layers: int = 18
    encoder_hidden: int = 1024
    encoder_heads: int = 16

    # VAEs
    video_vae_params_b: float = 0.4
    audio_vae_params_b: float = 0.2
    video_vae_spatial_downsample: int = 8
    video_vae_temporal_downsample: int = 4
    video_vae_channels: int = 16
    audio_vae_channels: int = 8
    audio_sample_rate_hz: int = 24000
    audio_frame_rate_hz: int = 75

    # Attention
    window_size: int = 16          # spatial windowed attention
    global_temporal_every: int = 4 # global temporal attention every N layers

    # Joint stream
    modality_embeddings: tuple[str, ...] = ("video", "audio", "text", "ref")


@dataclass(frozen=True)
class IOConfig:
    """Input/output limits per the Seedance 2.0 report (§1.1–1.2)."""

    max_text_prompt: int = 1
    max_images: int = 9
    max_videos: int = 3
    max_audio: int = 3

    min_duration_s: float = 4.0
    max_duration_s: float = 15.0
    resolutions: tuple[Resolution, ...] = (Resolution.P480, Resolution.P720)
    fps: int = 24

    audio_channels: int = 2        # binaural
    audio_tracks: tuple[str, ...] = ("bgm", "ambient", "dialogue")


@dataclass(frozen=True)
class InferenceConfig:
    """Sampling configuration."""

    variant: Variant = Variant.FULL
    full_steps: int = 50
    fast_steps: int = 6
    cfg_scale_text: float = 6.0
    cfg_scale_subject: float = 4.0
    cfg_scale_style: float = 3.0
    cfg_scale_motion: float = 3.5
    cfg_scale_audio: float = 4.0
    cfg_scale_vfx: float = 3.5
    seed: int | None = None


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    io: IOConfig = field(default_factory=IOConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    extra: dict[str, Any] = field(default_factory=dict)


DEFAULT_CONFIG = Config()


# --- Presets -----------------------------------------------------------------

# TINY: fits in ~6 GB VRAM (RTX 4060 8GB laptop). For smoke-testing the scaffold
# end-to-end only — NOT a production-quality model. ~0.5B core, 480p/4s, Fast.
TINY_MODEL = ModelConfig(
    core_layers=12,
    core_hidden=512,
    core_heads=8,
    core_params_b=0.5,
    ar_layers=4,
    ar_params_b=0.05,
    text_encoder_params_b=0.1,
    image_vit_layers=6,
    video_encoder_layers=6,
    audio_encoder_layers=6,
    encoder_hidden=256,
    encoder_heads=8,
    video_vae_channels=8,
    audio_vae_channels=4,
    window_size=8,
    global_temporal_every=3,
)

TINY_INFERENCE = InferenceConfig(variant=Variant.FAST, fast_steps=4)


def tiny_config() -> Config:
    """Build a Config that fits in ~6 GB VRAM for smoke-testing on consumer GPUs."""
    return Config(model=TINY_MODEL, io=IOConfig(), inference=TINY_INFERENCE)