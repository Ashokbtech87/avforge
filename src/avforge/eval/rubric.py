"""SeedVideoBench 2.0 rubric — category lists and target thresholds.

Re-exported from the eval package for clarity. See docs/04_EVALUATION.md.
"""

from . import (
    AUDIO_CATEGORIES,
    AUDIO_TARGETS,
    MOTION_CATEGORIES,
    MOTION_TARGETS,
    R2V_EXTENSION_TARGETS,
    R2V_TARGETS,
    SATISFACTION_TARGET,
    T2V_TARGETS,
    USABILITY_TARGET,
    CategoryScore,
    EvalReport,
    arena_preference,
    evaluate,
)

__all__ = [
    "AUDIO_CATEGORIES", "AUDIO_TARGETS", "MOTION_CATEGORIES", "MOTION_TARGETS",
    "R2V_TARGETS", "R2V_EXTENSION_TARGETS", "T2V_TARGETS",
    "USABILITY_TARGET", "SATISFACTION_TARGET",
    "CategoryScore", "EvalReport", "evaluate", "arena_preference",
]