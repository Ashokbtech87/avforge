"""Reference Registry — the controllability core.

Implements the typed-slot reference system from docs/02_ARCHITECTURE.md §2.
Each reference input (image/video/audio/style/vfx/first-frame/edit-instruction/source-clip)
is routed to a typed slot with a stable entity/speaker ID, and exposed as a
gated cross-attention KV cache consumed by the MMDiT backbone.

This module is the key to supporting the 20/22 R2V task matrix (Table 25) and
the 7 exclusive tasks (3 VFX/creative + 4 continuation/extension) that Seedance 2.0
lists as differentiators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import torch
from torch import Tensor


class SlotType(str, Enum):
    SUBJECT_IMAGE = "subject_image"
    SUBJECT_VIDEO = "subject_video"
    MOTION_VIDEO = "motion_video"
    STYLE_IMAGE = "style_image"
    STYLE_VIDEO = "style_video"
    VFX_REFERENCE = "vfx_reference"
    AUDIO_REFERENCE = "audio_reference"
    FIRST_FRAME = "first_frame"
    LAST_FRAME = "last_frame"
    EDIT_INSTRUCTION = "edit_instruction"
    SOURCE_CLIP = "source_clip"


@dataclass
class ReferenceSlot:
    """A single reference input routed to a typed slot.

    Attributes:
        slot_type: one of SlotType.
        tokens: encoded reference tokens (B, N, D) from the appropriate encoder.
        entity_id: stable per-entity ID for subject slots (multi-subject tracking).
        speaker_id: stable per-speaker ID for audio slots (multi-speaker lip-sync).
        role: optional role tag (e.g. "dialogue", "music", "ambient", "singing").
        preserve_mask: for source_clip slots, 1 = keep, 0 = may edit (editing tasks).
        gate: learned cross-attention gate, initialized near zero for stability.
    """

    slot_type: SlotType
    tokens: Tensor
    entity_id: int | None = None
    speaker_id: int | None = None
    role: str | None = None
    preserve_mask: Tensor | None = None
    gate: float = 0.0


@dataclass
class ReferenceRegistry:
    """Persistent KV cache of all reference slots for a generation request.

    The MMDiT backbone attends to all slots via gated cross-attention at every block.
    Subject-slot tracking (entity_id) prevents multi-subject omission/duplication,
    the reported weakness in Seedance 2.0 continuation.
    """

    slots: list[ReferenceSlot] = field(default_factory=list)
    _next_entity_id: int = 0
    _next_speaker_id: int = 0

    def add(self, slot: ReferenceSlot) -> None:
        # Auto-assign stable IDs for subject/audio slots if not provided.
        if slot.slot_type in (SlotType.SUBJECT_IMAGE, SlotType.SUBJECT_VIDEO):
            if slot.entity_id is None:
                slot.entity_id = self._next_entity_id
                self._next_entity_id += 1
        if slot.slot_type == SlotType.AUDIO_REFERENCE:
            if slot.speaker_id is None:
                slot.speaker_id = self._next_speaker_id
                self._next_speaker_id += 1
        self.slots.append(slot)

    def by_type(self, slot_type: SlotType) -> list[ReferenceSlot]:
        return [s for s in self.slots if s.slot_type == slot_type]

    def entity_count(self) -> int:
        """Number of distinct subject entities registered — used by the
        count-consistency reward to penalize multi-subject omission/duplication."""
        return len({s.entity_id for s in self.slots if s.entity_id is not None})

    def speaker_count(self) -> int:
        return len({s.speaker_id for s in self.slots if s.speaker_id is not None})

    def kv_cache(self) -> tuple[Tensor, Tensor, Tensor]:
        """Concatenate all slot tokens into a single KV cache + gates.

        Returns:
            keys:   (B, N_total, D)
            values: (B, N_total, D)
            gates:  (N_total,) per-token gate weights
        """
        if not self.slots:
            # Return a single zero-token cache so the unconditional/text-only pass
            # has a valid KV cache. (Real impl uses a learned null-token.)
            D = 512  # safe default; backbone projects anyway
            null = torch.zeros(1, 1, D, dtype=torch.float32)
            return null, null, torch.zeros(1, dtype=torch.float32)
        keys = torch.cat([s.tokens for s in self.slots], dim=1)
        values = keys  # symmetric for this reference impl
        gates = torch.tensor([s.gate for s in self.slots], dtype=keys.dtype)
        return keys, values, gates

    def describe(self) -> str:
        """Human-readable task description, used for logging and eval mode."""
        lines = [f"ReferenceRegistry: {len(self.slots)} slot(s), "
                 f"{self.entity_count()} subject(s), {self.speaker_count()} speaker(s)"]
        for s in self.slots:
            extra = []
            if s.entity_id is not None:
                extra.append(f"entity={s.entity_id}")
            if s.speaker_id is not None:
                extra.append(f"speaker={s.speaker_id}")
            if s.role is not None:
                extra.append(f"role={s.role}")
            if s.preserve_mask is not None:
                keep = int(s.preserve_mask.sum().item())
                total = int(s.preserve_mask.numel())
                extra.append(f"preserve={keep}/{total}")
            lines.append(f"  - {s.slot_type.value}: tokens={tuple(s.tokens.shape)} "
                         + " ".join(extra))
        return "\n".join(lines)


def build_registry(
    *,
    subject_images: Sequence[Tensor] | None = None,
    subject_videos: Sequence[Tensor] | None = None,
    motion_videos: Sequence[Tensor] | None = None,
    style_images: Sequence[Tensor] | None = None,
    style_videos: Sequence[Tensor] | None = None,
    vfx_references: Sequence[Tensor] | None = None,
    audio_references: Sequence[Tuple[Tensor, str]] | None = None,
    first_frame: Tensor | None = None,
    last_frame: Tensor | None = None,
    edit_instruction: Tensor | None = None,
    source_clip: tuple[Tensor, Tensor] | None = None,
    gate: float = 0.1,
) -> ReferenceRegistry:
    """Convenience builder mapping raw reference tensors to typed slots.

    `audio_references` is a sequence of (tokens, role) where role is one of
    "dialogue" | "music" | "ambient" | "singing".
    `source_clip` is (tokens, preserve_mask) for editing/continuation/extension.
    """
    reg = ReferenceRegistry()
    g = gate
    for t in subject_images or []:
        reg.add(ReferenceSlot(SlotType.SUBJECT_IMAGE, t, gate=g))
    for t in subject_videos or []:
        reg.add(ReferenceSlot(SlotType.SUBJECT_VIDEO, t, gate=g))
    for t in motion_videos or []:
        reg.add(ReferenceSlot(SlotType.MOTION_VIDEO, t, gate=g))
    for t in style_images or []:
        reg.add(ReferenceSlot(SlotType.STYLE_IMAGE, t, gate=g))
    for t in style_videos or []:
        reg.add(ReferenceSlot(SlotType.STYLE_VIDEO, t, gate=g))
    for t in vfx_references or []:
        reg.add(ReferenceSlot(SlotType.VFX_REFERENCE, t, gate=g))
    for t, role in audio_references or []:
        reg.add(ReferenceSlot(SlotType.AUDIO_REFERENCE, t, role=role, gate=g))
    if first_frame is not None:
        reg.add(ReferenceSlot(SlotType.FIRST_FRAME, first_frame, gate=g))
    if last_frame is not None:
        reg.add(ReferenceSlot(SlotType.LAST_FRAME, last_frame, gate=g))
    if edit_instruction is not None:
        reg.add(ReferenceSlot(SlotType.EDIT_INSTRUCTION, edit_instruction, gate=g))
    if source_clip is not None:
        tokens, mask = source_clip
        reg.add(ReferenceSlot(SlotType.SOURCE_CLIP, tokens, preserve_mask=mask, gate=g))
    return reg


# Tuple is used in type hints above; import here to avoid runtime issues in annotations.
from typing import Tuple  # noqa: E402