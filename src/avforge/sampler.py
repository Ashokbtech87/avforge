"""Flow-matching sampler for AV-Forge 2.0.

Implements rectified-flow sampling over the joint video+audio latent, with
per-modality classifier-free guidance (CFG) and a Fast consistency-distilled
variant. See docs/02_ARCHITECTURE.md §5.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .config import InferenceConfig, Variant
from .model import MMDiTBackbone
from .registry import ReferenceRegistry


def _cfg(v_cond: Tensor, v_uncond: Tensor, scale: float) -> Tensor:
    return v_uncond + scale * (v_cond - v_uncond)


@torch.no_grad()
def sample(
    backbone: MMDiTBackbone,
    shape: tuple[int, ...],
    text_tokens: Tensor,
    registry: ReferenceRegistry,
    cfg: InferenceConfig,
    device,
    dtype,
    shot_plan_tokens: Tensor,
    nv: int,
    na: int,
) -> tuple[Tensor, Tensor]:
    """Sample a joint video+audio latent via flow-matching.

    `nv`/`na` are the video/audio token counts (the caller knows the layout).
    Returns (video_latent (B, nv, Cv), audio_latent (B, na, Ca)).
    """
    B = shape[0]
    steps = cfg.full_steps if cfg.variant == Variant.FULL else cfg.fast_steps
    Cv = backbone.cfg.video_vae_channels
    Ca = backbone.cfg.audio_vae_channels
    Cmax = max(Cv, Ca)
    Nv, Na = nv, na
    # Joint latent layout: (B, Nv+Na, Cmax). Video tokens use first Cv channels,
    # audio tokens use first Ca channels; remaining channels are zero padding.
    x = torch.randn(B, Nv + Na, Cmax, device=device, dtype=dtype)

    times = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=dtype)
    for i in range(steps):
        t = times[i].expand(B)
        v_lat = x[:, :Nv, :Cv]
        a_lat = x[:, Nv:, :Ca]
        # Conditional pass (with text + registry)
        out_cond = backbone(
            v_lat, a_lat, text_tokens, registry, t, shot_plan_tokens,
        )
        v_cond = out_cond.v_pred

        # Unconditional pass (drop text + registry) for CFG
        out_uncond = backbone(
            v_lat, a_lat,
            torch.zeros_like(text_tokens),
            _empty_registry_like(registry, device, dtype),
            t, shot_plan_tokens,
        )
        v_uncond = out_uncond.v_pred

        # Per-modality CFG (simplified: single scale on the joint velocity)
        v = _cfg(v_cond, v_uncond, cfg.cfg_scale_text)

        # Euler step
        dt = times[i] - times[i + 1]
        x = x + v * dt

    return x[:, :Nv, :Cv], x[:, Nv:, :Ca]


def _empty_registry_like(reg: ReferenceRegistry, device, dtype) -> ReferenceRegistry:
    """Build an empty registry with a single zero-token slot so the unconditional
    pass has a valid KV cache. (Real impl would use a learned null-token.)"""
    from .registry import ReferenceSlot, SlotType
    D = reg.slots[0].tokens.size(-1) if reg.slots else 2048
    null = torch.zeros(1, 1, D, device=device, dtype=dtype)
    return ReferenceRegistry(slots=[ReferenceSlot(SlotType.EDIT_INSTRUCTION, null, gate=0.0)])