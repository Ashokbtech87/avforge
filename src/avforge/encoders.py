"""Encoders for text, image, video, and audio reference inputs.

These are thin reference implementations of the encoder spec in
docs/02_ARCHITECTURE.md §1.3. In a real deployment these would be the actual
pretrained encoder weights; here they are shape-correct stubs so the pipeline
runs end-to-end and the architecture is concrete.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import ModelConfig


class TextEncoder(nn.Module):
    """7B LLM text encoder (frozen). Produces (B, N_text, D) tokens + pooled summary."""

    def __init__(self, cfg: ModelConfig, vocab: int = 128000):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(vocab, cfg.core_hidden)
        self.proj = nn.Linear(cfg.core_hidden, cfg.core_hidden)

    def forward(self, token_ids: Tensor) -> tuple[Tensor, Tensor]:
        x = self.embed(token_ids)
        x = self.proj(x)
        pooled = x.mean(dim=1, keepdim=True)
        return x, pooled


class ImageViTEncoder(nn.Module):
    """ViT-style image encoder for subject/style/first-frame references."""

    def __init__(self, cfg: ModelConfig, patch: int = 16):
        super().__init__()
        self.cfg = cfg
        self.patch = patch
        self.patch_embed = nn.Conv2d(3, cfg.encoder_hidden, kernel_size=patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, 1024, cfg.encoder_hidden))
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                cfg.encoder_hidden, cfg.encoder_heads, cfg.encoder_hidden * 4,
                batch_first=True, norm_first=True,
            )
            for _ in range(cfg.image_vit_layers)
        ])
        self.proj = nn.Linear(cfg.encoder_hidden, cfg.core_hidden)

    def forward(self, image: Tensor) -> Tensor:
        # image: (B, 3, H, W)
        x = self.patch_embed(image)              # (B, D, h, w)
        B, D, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)         # (B, N, D)
        x = x + self.pos[:, : x.size(1)]
        for blk in self.blocks:
            x = blk(x)
        return self.proj(x)


class VideoEncoder(nn.Module):
    """Spatiotemporal encoder for subject/motion/style/vfx video references."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = nn.Conv3d(3, cfg.encoder_hidden, kernel_size=(2, 16, 16),
                                    stride=(2, 16, 16))
        self.pos = nn.Parameter(torch.zeros(1, 4096, cfg.encoder_hidden))
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                cfg.encoder_hidden, cfg.encoder_heads, cfg.encoder_hidden * 4,
                batch_first=True, norm_first=True,
            )
            for _ in range(cfg.video_encoder_layers)
        ])
        self.proj = nn.Linear(cfg.encoder_hidden, cfg.core_hidden)

    def forward(self, video: Tensor) -> Tensor:
        # video: (B, C, T, H, W)
        x = self.patch_embed(video)
        B, D, t, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos[:, : x.size(1)]
        for blk in self.blocks:
            x = blk(x)
        return self.proj(x)


class AudioEncoder(nn.Module):
    """Audio encoder for dialogue/music/ambient/singing references.

    Operates on mel-spectrograms at 24 kHz. Produces tokens tagged with role.
    """

    def __init__(self, cfg: ModelConfig, n_mels: int = 128):
        super().__init__()
        self.cfg = cfg
        self.mel_proj = nn.Linear(n_mels, cfg.encoder_hidden)
        self.pos = nn.Parameter(torch.zeros(1, 4096, cfg.encoder_hidden))
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                cfg.encoder_hidden, cfg.encoder_heads, cfg.encoder_hidden * 4,
                batch_first=True, norm_first=True,
            )
            for _ in range(cfg.audio_encoder_layers)
        ])
        self.proj = nn.Linear(cfg.encoder_hidden, cfg.core_hidden)

    def forward(self, mel: Tensor) -> Tensor:
        # mel: (B, N_frames, n_mels)
        x = self.mel_proj(mel)
        x = x + self.pos[:, : x.size(1)]
        for blk in self.blocks:
            x = blk(x)
        return self.proj(x)


class VideoVAE(nn.Module):
    """3D causal VAE: pixels <-> video latent (T×H×W×C, 4×/8×/8× downsample)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        C = cfg.video_vae_channels
        self.encoder = nn.Conv3d(3, C, kernel_size=(4, 8, 8), stride=(4, 8, 8), padding=(1, 3, 3))
        self.decoder = nn.ConvTranspose3d(C, 3, kernel_size=(4, 8, 8), stride=(4, 8, 8), padding=(1, 3, 3))

    def encode(self, video: Tensor) -> Tensor:
        return self.encoder(video)

    def decode(self, latent: Tensor) -> Tensor:
        return self.decoder(latent)


class AudioVAE(nn.Module):
    """AudioVAE: waveform <-> audio latent (75 Hz, 8 channels, binaural)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        C = cfg.audio_vae_channels
        self.encoder = nn.Conv1d(2, C, kernel_size=320, stride=320)
        self.decoder = nn.ConvTranspose1d(C, 2, kernel_size=320, stride=320)

    def encode(self, waveform: Tensor) -> Tensor:
        # waveform: (B, 2, samples)
        return self.encoder(waveform)

    def decode(self, latent: Tensor) -> Tensor:
        return self.decoder(latent)


def build_encoders(cfg: ModelConfig) -> dict[str, nn.Module]:
    return {
        "text": TextEncoder(cfg),
        "image": ImageViTEncoder(cfg),
        "video": VideoEncoder(cfg),
        "audio": AudioEncoder(cfg),
        "video_vae": VideoVAE(cfg),
        "audio_vae": AudioVAE(cfg),
    }