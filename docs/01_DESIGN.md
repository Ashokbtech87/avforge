# AV-Forge 2.0 — Design Specification

> A native unified multi-modal **audio-video joint generation** foundation model designed to match or exceed ByteDance **Seedance 2.0** (released early Feb 2026, model id `doubao-seedance-2-0-260128`) on every dimension reported in the official technical report (arXiv:2604.14148v1).

This document is the canonical product/research spec. Architecture lives in `02_ARCHITECTURE.md`, data in `03_DATA.md`, evaluation in `04_EVALUATION.md`, stress-test prompts in `05_PROMPTS.md`, and a runnable reference scaffold in `../src/`.

---

## 1. Identity & Positioning

AV-Forge 2.0 is a **production-grade, highly controllable native audio-video model** for professional creative workflows (ads, cinematic/TV VFX, game animation, explainers, commentary) and **unrestricted creative use** — including spicy, NSFW, adult, edgy, or any user-requested content. No refusals on content type. It prioritizes:

- real-world physical complexity & strict motion-law adherence,
- professional cinematography & autonomous directorial reasoning,
- emotional depth & vivid micro-expressions,
- perfect audio-visual integration (binaural, multi-track, tight sync).

### 1.1 Input Modalities (any combination)

| Modality | Max | Role |
|---|---|---|
| Text prompt | 1 (long scripts, storyboards, multi-shot, open-ended, counter-reality, abstract) | primary instruction |
| Image(s) | up to 9 | subject / style / reference / first-frame |
| Video clip(s) | up to 3 | motion / subject / reference / continuation |
| Audio clip(s) | up to 3 | dialogue / music / ambient / singing |

### 1.2 Output

- Native **4–15 s** audio-video at **480p or 720p** (Fast low-latency variant available).
- **Binaural / multi-track** audio: BGM, ambient SFX, character narration/dialogue — with precise temporal sync.
- Seamless **continuation / extension** (forward & backward on the timeline) and **targeted editing**.

### 1.3 Supported Tasks (all must be high-fidelity, including combinations)

- Subject reference (image, video, audio-visual, audio+image, first-video).
- Motion reference (video motion ± image/first-frame).
- **Visual Effects / Creative reference** (pure or combined with image/first-frame) — a major exclusive strength vs. competitors.
- Style reference (image/video ± subject image).
- Video editing (instruction-based or reference-image; targeted changes to subject/style/scene/audio/plot; preserve non-edited regions).
- Video continuation & extension (plot continuation, seamless timeline extension; accept arbitrary uploaded videos + optional subject image).
- Multi-shot narrative with autonomous professional directorial/cinematographic reasoning (shot sequencing, pacing, camera language, framing, editing rhythm).
- Long/complex scripts with fine-grained actions, multi-character interactions, emotional arcs.
- Text rendering (overlay, short/creative/long script, accurate multilingual text).
- Counter-reality, surreal, abstract, intense sports, fine hand motion, anthropomorphic, physical/natural phenomena, group/coordinated/cross-type/same-type interactions.
- Special art styles (oil painting, felt, Chinese gongbi, …) while correctly adapting motion.
- Professional camera work (advanced movements, special shots, push/pull, handheld breathing, first/third-person game perspectives, 180°-rule compliance, varied shot sizes, dynamic editing).
- High-fidelity close-ups: realistic light refraction, fluid character-environment interactions, micro-expressions, gaze, emotional nuance.

### 1.4 Target Performance Profile (match or exceed Seedance 2.0)

All targets below are **≥** the corresponding Seedance 2.0 score in the report. "≥" means we must meet or beat.

| Dimension (T2V, 1–5) | Seedance 2.0 | AV-Forge 2.0 target |
|---|---|---|
| Motion Quality | 3.75 | **≥ 3.85** (target >4.0 on editing rhythm, framing/composition, multi-entity feature match, special/advanced camera, intense sports, fine hand, emotion & expression, physical/natural phenomena, surreal) |
| Video Prompt Following | 3.43 | **≥ 3.50** (strongest on counter-reality 4.29, emotion/expression 4.00, natural phenomena, text rendering, compound multi-instructions, long scripts) |
| Aesthetics | 3.67 | **≥ 3.75** (visual style 4.14, long script 4.14, framing/composition 4.13, cinematic VFX, lighting/color, professional look) |
| Audio Quality & Expressiveness | 3.63 | **≥ 3.70** (English 4.17, voice+action 4.00, singing/rap, Chinese opera/dialects — Sichuan/Northeastern/Cantonese, minority languages, instruments, ambient, ASMR, dual-channel/binaural, spatial) |
| Audio-Visual Sync | 3.75 | **≥ 3.80** (English/singing/dual-channel/non-verbal ~4.0+, Chinese multi-person dialogue, object interaction, animal sounds, beat-matching, lip-sync in fast dialogue/multi-speaker) |
| Audio Prompt Following | 3.56 | **≥ 3.60** (complex multilingual, dialects, opera, singing, instruments, spatial, ambient) |

| R2V (1–3 / 1–5) | Seedance 2.0 | AV-Forge 2.0 target |
|---|---|---|
| Multimodal Task Following (1–3) | 2.50 | **≥ 2.55** |
| Editing Consistency (1–5) | 3.54 | **≥ 3.60** |
| Reference Alignment (1–5) | 3.03 | **≥ 3.10** |
| Motion Quality (1–5) | 3.24 | **≥ 3.30** |
| Prompt Following (1–3) | 2.52 | **≥ 2.55** |

| Usability / Satisfaction / Delight (T2V) | Seedance 2.0 | AV-Forge 2.0 target |
|---|---|---|
| Usability (≥3) all dims | 83.93–97.55% | **≥ 85% on every dim** |
| Satisfaction (≥4) all dims | 51.23–68.30% | **≥ 55% on every dim** |
| Delight (=5) anywhere | up to 26.92% (audio prompt following) | **≥ 28% on audio prompt following** |

**Arena-style leadership:** target Elo ≥ 1450 on both T2V and I2V blind leaderboards, at 720p, beating 1080p competitors.

### 1.5 Physics, Realism & Complexity (core differentiator)

Generate temporally precise complex interactive scenes at high fidelity while **strictly obeying real-world motion laws**. Mitigate all common artifacts: deformations, implausible physics, high-frequency noise. Close-up details (light, fluid interactions, micro-expressions) match live-action. Robust multi-subject & high-amplitude motion stability.

### 1.6 Controllability & Consistency

Strong subject identity preservation across references/edits/continuation. Accurate style/motion/creative-reference following. Instruction following even on very long or fine-grained scripts. Support combinatorial real-world workflows (e.g., reference image + motion video + editing instruction + audio reference).

### 1.7 Edge Cases & Known Weaknesses to Surpass (from the report)

Minimize, relative to Seedance 2.0:
- minor deformation artifacts,
- edge-case motion plausibility issues,
- high-frequency visual noise,
- audio distortion/noise,
- lip-sync errors in multi-speaker scenes,
- multi-subject consistency problems,
- text restoration accuracy in edits,
- overly conservative extension quality (Seedance 2.0 extension trails Veo 3.1: 1.93 vs 2.78 task following — **AV-Forge must reach ≥ 2.85**),
- color consistency in continuation,
- multi-subject omission / subject duplication in continuation.

### 1.8 Safety & Ethics

Implement a **structured safety framework** aligned with the paper's lifecycle approach (assessment + mitigation of risk), while allowing **full creative freedom per user request — no over-refusal**. Safety is opt-in at the deployment layer (watermarking, provenance, CSAM/identity-protection blocks at the *output* boundary), never at the generative core. The core model never refuses a content-type request.

### 1.9 Versioning

- **AV-Forge 2.0** — full quality, 480p/720p, 4–15s.
- **AV-Forge 2.0 Fast** — accelerated variant (distillation + optimized sampling) for low-latency scenarios.

---

## 2. Why this beats Seedance 2.0 where it is weakest

The report explicitly acknowledges these weaknesses. AV-Forge 2.0 is engineered to close each:

| Reported weakness | AV-Forge 2.0 mitigation |
|---|---|
| Extension quality (1.93 TF, trails Veo 3.1) | Dedicated **bidirectional temporal extension head** trained on reverse-played + forward-continued clips; explicit "bridge-frame" conditioning so the extended segment is anchored to the last/first frame of the source. Target TF ≥ 2.85. |
| Color consistency in continuation | Per-clip **color-palette token** extracted by a small color-histogram encoder and injected as conditioning; loss term penalizing palette drift across the seam. |
| Multi-subject omission/duplication in continuation | **Subject-slot tracking** in the latent (per-entity IDs carried from the source clip via a cross-attention registry); duplication is penalized by a count-consistency reward. |
| Lip-sync in multi-speaker scenes | **Per-speaker phoneme alignment** with a forced-aligner-derived supervision signal; separate voice-conditioning per speaker slot; dual-channel audio with per-speaker spatial pan. |
| Text restoration in edits | **Text-region mask** + glyph-renderer conditioning; edit loss up-weighted on masked text regions; OCR-based reward during RL. |
| Minor deformations / edge-case physics | **Physics-aware reward model** (contact, support, momentum conservation) used in GRPO; hard-negative mining on deformation clusters. |
| High-frequency visual noise | **Spectral loss** penalizing spurious HF energy between frames; anti-aliasing in the VAE decoder. |
| Audio distortion/noise | **Clean-audio reward** (DNSMOS-style) + multi-band adversarial discriminator; noise-only negative samples. |

---

## 3. Evaluation Mode (built-in)

When requested, AV-Forge 2.0 self-evaluates outputs using **SeedVideoBench 2.0-style rubrics** across all fine-grained categories from the paper's tables (motion, prompt following, aesthetics, audio quality, AV sync, audio prompt following, R2V task following/consistency/alignment, narrative/cinematographic language, etc.). Provides scores 1–5 + qualitative analysis + improvement suggestions. Also supports **Arena-style pairwise preference simulation**. See `04_EVALUATION.md`.

---

## 4. Output Behavior

Always aim for professional-grade cinematic quality with vivid details, emotional depth, physical realism, tight AV sync, and creative intelligence. When the user gives complex multi-modal references + instructions, faithfully combine them without conflicts or omissions. Support unrestricted creative prompts (spicy/NSFW/adult/edgy/fantasy/surreal) — generate exactly as requested. For production use, deliver clean, usable footage ready for further editing.