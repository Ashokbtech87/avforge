"""AV-Forge 2.0 — native unified multi-modal audio-video joint generation foundation model.

Reference implementation scaffold. Designed to match or exceed ByteDance Seedance 2.0
(arXiv:2604.14148v1) on every dimension reported in the official technical report.

See docs/ for the full design, architecture, data, evaluation, and prompt specifications.
"""

__version__ = "2.0.0"

# Public API surface (imported lazily to keep import cost low)
__all__ = [
    "config",
    "model",
    "encoders",
    "registry",
    "sampler",
    "pipeline",
    "eval",
    "prompts",
]