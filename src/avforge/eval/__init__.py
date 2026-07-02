"""SeedVideoBench 2.0-style evaluation rubric.

Encodes the exact fine-grained categories and target thresholds from the
Seedance 2.0 report (Tables 3–8, 11–23, 26–28) so AV-Forge 2.0's self-evaluation
mode is reproducible. See docs/04_EVALUATION.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# --- Fine-grained category lists (verbatim from the report) ---

MOTION_CATEGORIES: tuple[str, ...] = (
    "Holidays / Festivals", "Consumer Visual Effects", "Counter-Reality Instructions",
    "Cinematic Visual Effects", "Same-Type Interaction", "Cross-Type Interaction",
    "Group Coordinated Motion", "Advanced Camera Movement", "Special Camera Shots",
    "Editing Rhythm", "Combined Shot Instructions", "Physical Feedback",
    "Physical Phenomena", "Natural Phenomena", "Text Overlay", "Short Text",
    "Creative Text", "Long Script", "Abstract Challenges", "Multi-Entity Feature Match",
    "Knowledge Assessment", "Compound Multi-Instructions", "Surreal Motion",
    "Intense Sports Motion", "Fine Hand Motion", "Anthropomorphic Motion",
    "Emotion & Expression", "Visual Style", "Lighting & Color Tone", "Framing / Composition",
)

AUDIO_CATEGORIES: tuple[str, ...] = (
    "Chinese Dialect / Accent", "Chinese Multi-Person Dialogue", "Chinese Variety Show Voice",
    "Chinese Opera", "English", "Minority Languages", "Singing / Rap", "Spatial Scene",
    "Off-Screen Voice", "Non-Verbal Voice", "Voice + Action Interaction",
    "Object Interaction Sound", "Animal Sound", "Ambient / Background Sound",
    "Special Effects (ASMR, etc.)", "Instruments & Audio", "Dual-Channel Audio",
)

# --- Target thresholds (≥ Seedance 2.0 scores) ---

# T2V overall (1–5)
T2V_TARGETS = {
    "Motion Quality": 3.85,
    "Video Prompt Following": 3.50,
    "Aesthetics": 3.75,
    "Audio Quality": 3.70,
    "Audio-Visual Sync": 3.80,
    "Audio Prompt Following": 3.60,
}

# T2V motion fine-grained targets (≥ Seedance 2.0 Table 3)
MOTION_TARGETS = {
    "Multi-Entity Feature Match": 4.43, "Framing / Composition": 4.25,
    "Editing Rhythm": 4.21, "Special Camera Shots": 3.92,
    "Advanced Camera Movement": 3.77, "Intense Sports Motion": 3.79,
    "Fine Hand Motion": 3.71, "Emotion & Expression": 4.00,
    "Natural Phenomena": 3.78, "Physical Phenomena": 3.38,
    "Physical Feedback": 3.46, "Surreal Motion": 3.71,
    "Counter-Reality Instructions": 3.71, "Same-Type Interaction": 3.79,
    "Cross-Type Interaction": 3.57, "Group Coordinated Motion": 3.29,
    "Text Overlay": 3.69, "Short Text": 3.71, "Creative Text": 3.57,
    "Long Script": 3.57, "Abstract Challenges": 4.00,
    "Compound Multi-Instructions": 3.71, "Anthropomorphic Motion": 3.29,
    "Knowledge Assessment": 3.69, "Visual Style": 4.00,
    "Lighting & Color Tone": 3.71, "Combined Shot Instructions": 3.86,
    "Holidays / Festivals": 3.29, "Consumer Visual Effects": 3.71,
    "Cinematic Visual Effects": 3.79,
}

# T2V audio fine-grained targets (≥ Tables 6/7/8)
AUDIO_TARGETS = {
    "English": 4.17, "Voice + Action Interaction": 4.00,
    "Minority Languages": 3.82, "Ambient / Background Sound": 3.78,
    "Object Interaction Sound": 3.76, "Singing / Rap": 3.71,
    "Animal Sound": 3.57, "Special Effects (ASMR, etc.)": 3.71,
    "Instruments & Audio": 3.68, "Chinese Dialect / Accent": 2.82,
    "Chinese Opera": 3.75, "Chinese Multi-Person Dialogue": 3.71,
    "Chinese Variety Show Voice": 3.14, "Spatial Scene": 3.43,
    "Off-Screen Voice": 3.29, "Non-Verbal Voice": 3.56,
    "Dual-Channel Audio": 3.43,
}

# R2V targets (Table 24)
R2V_TARGETS = {
    "Multimodal Task Following": 2.55,   # 1–3 scale
    "Editing Consistency": 3.60,         # 1–5
    "Reference Alignment": 3.10,         # 1–5
    "Motion Quality": 3.30,              # 1–5
    "Prompt Following": 2.55,            # 1–3
}

# R2V extension (Seedance 2.0's weakest task: TF 1.93 — AV-Forge must reach ≥2.85)
R2V_EXTENSION_TARGETS = {"Task Following": 2.85, "Reference Alignment": 3.30}

# Usability / Satisfaction / Delight thresholds
USABILITY_TARGET = 0.85   # ≥85% on every dim
SATISFACTION_TARGET = 0.55
DELIGHT_TARGETS = {"Audio Prompt Following": 0.28}


@dataclass
class CategoryScore:
    category: str
    score: float
    target: float
    passes: bool
    margin: float

    def __post_init__(self):
        self.passes = self.score >= self.target
        self.margin = self.score - self.target


@dataclass
class EvalReport:
    task: str
    duration_s: float
    resolution: str
    variant: str
    overall: dict[str, CategoryScore] = field(default_factory=dict)
    fine_grained: dict[str, CategoryScore] = field(default_factory=dict)
    narrative: dict[str, float] = field(default_factory=dict)
    usability: dict[str, float] = field(default_factory=dict)
    arena_elo: dict[str, int] = field(default_factory=dict)
    qualitative: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [
            f"AV-Forge 2.0 — SeedVideoBench 2.0 Evaluation",
            f"============================================",
            f"Task: {self.task} | Duration: {self.duration_s}s | "
            f"Resolution: {self.resolution} | Variant: {self.variant}",
            "",
            "Overall (1–5):",
        ]
        for name, cs in self.overall.items():
            mark = "✓" if cs.passes else "✗"
            lines.append(f"  {name}: {cs.score:.2f}  (target ≥{cs.target:.2f})  {mark}")
        lines.append("")
        lines.append("Fine-grained (top 5 / bottom 5):")
        ranked = sorted(self.fine_grained.values(), key=lambda c: c.margin)
        lines.append("  Bottom (closest to target):")
        for cs in ranked[:5]:
            lines.append(f"    {cs.category}: {cs.score:.2f} (target >{cs.target:.2f}) "
                         f"{'✓' if cs.passes else '✗'} (margin {cs.margin:+.2f})")
        lines.append("  Top (strongest):")
        for cs in sorted(self.fine_grained.values(), key=lambda c: -c.score)[:5]:
            lines.append(f"    {cs.category}: {cs.score:.2f} (target >{cs.target:.2f}) ✓")
        if self.narrative:
            lines.append("")
            lines.append("Narrative Quality (1–5):")
            for k, v in self.narrative.items():
                lines.append(f"  {k}: {v:.2f}")
        if self.usability:
            lines.append("")
            lines.append("Usability/Satisfaction/Delight:")
            for k, v in self.usability.items():
                lines.append(f"  {k}: {v}")
        if self.arena_elo:
            lines.append("")
            lines.append("Arena simulation:")
            for k, v in self.arena_elo.items():
                lines.append(f"  {k}: {v}")
        if self.qualitative:
            lines.append("")
            lines.append("Qualitative analysis:")
            for q in self.qualitative:
                lines.append(f"  - {q}")
        return "\n".join(lines)


def evaluate(
    scores: dict[str, float],
    *,
    task: str = "T2V",
    duration_s: float = 10.0,
    resolution: str = "720p",
    variant: str = "full",
    fine_grained: dict[str, float] | None = None,
    narrative: dict[str, float] | None = None,
    usability: dict[str, float] | None = None,
    arena_elo: dict[str, int] | None = None,
    qualitative: list[str] | None = None,
) -> EvalReport:
    """Build an EvalReport from raw scores, checking against Seedance 2.0 targets.

    `scores` keys are overall dimensions (Motion Quality, etc.).
    `fine_grained` keys are fine-grained categories (Motion + Audio).
    """
    rep = EvalReport(task=task, duration_s=duration_s, resolution=resolution, variant=variant)
    targets = T2V_TARGETS if task in ("T2V", "I2V") else R2V_TARGETS
    for name, score in scores.items():
        tgt = targets.get(name, 0.0)
        rep.overall[name] = CategoryScore(name, score, tgt, False, 0.0)
    if fine_grained:
        all_targets = {**MOTION_TARGETS, **AUDIO_TARGETS}
        for name, score in fine_grained.items():
            tgt = all_targets.get(name, 0.0)
            rep.fine_grained[name] = CategoryScore(name, score, tgt, False, 0.0)
    rep.narrative = narrative or {}
    rep.usability = usability or {}
    rep.arena_elo = arena_elo or {}
    rep.qualitative = qualitative or []
    return rep


# Arena-style pairwise preference simulation
def arena_preference(rating_a: int, rating_b: int, k: int = 32) -> tuple[int, int]:
    """Simulate one Arena pairwise vote and return updated Elo (A, B).

    A wins with probability sigmoid(rating_a - rating_b) (holistic preference
    modeled by the human-preference reward used in training).
    """
    import math, random
    expected_a = 1.0 / (1.0 + math.exp(rating_b - rating_a))
    a_wins = random.random() < expected_a
    score_a, score_b = (1.0, 0.0) if a_wins else (0.0, 1.0)
    new_a = rating_a + k * (score_a - expected_a)
    new_b = rating_b + k * (score_b - (1 - expected_a))
    return round(new_a), round(new_b)