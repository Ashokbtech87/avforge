# AV-Forge 2.0 — Evaluation (SeedVideoBench 2.0-style)

This document operationalizes the evaluation framework from the Seedance 2.0 report (§2.2 SeedVideoBench 2.0) into a runnable rubric. It is used both for **self-evaluation mode** (built into the model's eval command) and for **Arena-style pairwise preference simulation**.

The rubric mirrors the report's structure exactly: T2V (6 dims × 30 fine categories), I2V (6 dims), R2V (5 dims + 22 task matrix), plus the narrative-quality module (cinematographic language, plot design, stylistic aesthetics), plus usability/satisfaction/delight thresholds.

---

## 1. Dimensions & scales

| Dimension | Scale | Source table |
|---|---|---|
| Motion Quality | 1–5 | T2V Table 3, I2V Tables 11–18, R2V Table 24 |
| Video Prompt Following | 1–5 | T2V Table 4, I2V, R2V |
| Aesthetics | 1–5 | T2V Table 5 |
| Audio Quality & Expressiveness | 1–5 | T2V Table 6, I2V Tables 19–23 |
| Audio-Visual Sync | 1–5 | T2V Table 7, I2V |
| Audio Prompt Following | 1–5 | T2V Table 8, I2V |
| Image Preservation (I2V) | 1–5 | I2V Table 9 |
| Multimodal Task Following (R2V) | 1–3 | R2V Table 24 |
| Editing Consistency (R2V) | 1–5 | R2V Table 24/28 |
| Reference Alignment (R2V) | 1–5 | R2V Tables 26–28 |
| Prompt Following (R2V) | 1–3 | R2V Table 24 |

### 1.1 Narrative quality (subjective, 3 sub-dims, 1–5)
- **Cinematographic language** — shot logic/expressiveness; flags redundant coverage, axis-crossing (180° rule), mismatched shot sizes, uneven pacing.
- **Plot design** — from vague/brief prompt → coherent + engaging.
- **Stylistic aesthetics** — lighting, framing, composition, color grading; character/costume/prop/set coherence.

---

## 2. Fine-grained category list (T2V Motion/Prompt/Aesthetics share these 30)

Holidays/Festivals, Consumer Visual Effects, Counter-Reality Instructions, Cinematic Visual Effects, Same-Type Interaction, Cross-Type Interaction, Group Coordinated Motion, Advanced Camera Movement, Special Camera Shots, Editing Rhythm, Combined Shot Instructions, Physical Feedback, Physical Phenomena, Natural Phenomena, Text Overlay, Short Text, Creative Text, Long Script, Abstract Challenges, Multi-Entity Feature Match, Knowledge Assessment, Compound Multi-Instructions, Surreal Motion, Intense Sports Motion, Fine Hand Motion, Anthropomorphic Motion, Emotion & Expression, Visual Style, Lighting & Color Tone, Framing/Composition.

## 3. Fine-grained audio categories (17)

Chinese Dialect/Acent, Chinese Multi-Person Dialogue, Chinese Variety Show Voice, Chinese Opera, English, Minority Languages, Singing/Rap, Spatial Scene, Off-Screen Voice, Non-Verbal Voice, Voice+Action Interaction, Object Interaction Sound, Animal Sound, Ambient/Background Sound, Special Effects (ASMR), Instruments & Audio, Dual-Channel Audio.

## 4. R2V task matrix (22 tasks, Table 25)

| Task | Input modality |
|---|---|
| Subject Reference | Image / Video / Audio-Visual / Audio+Image |
| Motion Reference | Video / Video+Image / Video+First-Frame |
| VFX/Creative Reference | Pure / +Image / +First-Frame |
| Style Reference | Style Image / Style Image+Subject / Style Video / Style Video+Subject |
| Video Editing | Instruction / Reference-Image |
| Continuation/Extension | Continuation / +Subject / Extension / +Subject |

AV-Forge 2.0 must support **20/22** (matching Seedance 2.0) and target the 7 exclusive tasks (3 VFX + 4 continuation/extension) as differentiators.

---

## 5. Scoring protocol

### 5.1 Objective track (automated)
- Motion stability: optical-flow + keypoint-consistency pipeline.
- AV sync: forced-aligner + SyncNet confidence.
- Audio quality: DNSMOS + multi-band spectral descriptors.
- Text rendering: OCR accuracy on text regions.
- Subject identity: ArcFace/voice embedding cosine to reference.
- Physics: contact/support/momentum estimator.
- Reference alignment: per-slot embedding similarity.
- Editing consistency: masked-region pixel/feature preservation.

### 5.2 Subjective track (blind expert review)
- Aesthetics, narrative quality, audio expressiveness, cinematographic language — blind 1–5 from expert evaluators (ad/game/VFX pros, per the report).
- Minimum evaluator count per sample: 3; report mean + std.

### 5.3 Realism study (per report §2.2.1)
Evaluators attempt to distinguish AV-Forge outputs from real clips; results feed back into aesthetic tuning.

---

## 6. Usability / Satisfaction / Delight thresholds (targets)

| Metric | Definition | AV-Forge 2.0 target |
|---|---|---|
| Usability | % samples score ≥ 3 | ≥ 85% on every dim |
| Satisfaction | % samples score ≥ 4 | ≥ 55% on every dim |
| Delight | % samples score = 5 | ≥ 28% on audio prompt following |

---

## 7. Arena-style pairwise preference simulation

Given two outputs A (AV-Forge 2.0) and B (competitor), produce:
- **Elo update** with K=32, expected from logistic of rating diff.
- **Per-dimension preference** (which model wins on each of the 6 T2V / 6 I2V / 5 R2V dims).
- **Holistic preference** (single vote) — simulated via the human-preference model used in training reward.
- **Rank spread** (consistency of top rank across dims).

Target: Elo ≥ 1450 on T2V and I2V at 720p, beating 1080p competitors (mirroring the report's Arena.AI result).

---

## 8. Self-evaluation output format

When the user invokes eval mode, AV-Forge 2.0 returns:

```
AV-Forge 2.0 — SeedVideoBench 2.0 Evaluation
============================================
Task: T2V | Duration: 12s | Resolution: 720p | Variant: full

Overall (1–5):
  Motion Quality:           3.88  (target ≥3.85)  ✓
  Video Prompt Following:   3.52  (target ≥3.50)  ✓
  Aesthetics:               3.78  (target ≥3.75)  ✓
  Audio Quality:            3.71  (target ≥3.70)  ✓
  Audio-Visual Sync:        3.82  (target ≥3.80)  ✓
  Audio Prompt Following:   3.61  (target ≥3.60)  ✓

Fine-grained (top 5 / bottom 5):
  Top:
    Multi-Entity Feature Match: 4.46  (target >4.43) ✓
    Framing/Composition:        4.28  (target >4.25) ✓
    Editing Rhythm:             4.24  (target >4.21) ✓
    Emotion & Expression:       4.05  (target >4.00) ✓
    Counter-Reality:            4.31  (target >4.29) ✓
  Bottom:
    Surreal Motion:             3.74  (target >3.71) ✓ (margin 0.03 — add hard negs)
    Group Coordinated Motion:    3.33  (target >3.29) ✓ (margin 0.04)
    ...

Narrative Quality (1–5):
  Cinematographic Language: 4.10
  Plot Design:               3.95
  Stylistic Aesthetics:      4.05

Usability/Satisfaction/Delight:
  Audio Prompt Following — usability 86%, satisfaction 60%, delight 29%  ✓

Qualitative analysis:
  - Strong: multi-entity identity preservation, binaural layering, beat-matched cuts.
  - Weak: surreal-motion margin thin; add 20k procedural surreal hard negs.
  - Suggestion: up-weight spectral loss on high-motion segments.

Arena simulation (vs competitor X):
  Elo: 1458 (T2V), 1455 (I2V) — rank 1 on both.
```

The exact rubric, category list, and target thresholds are encoded in `src/avforge/eval/rubric.py` so eval mode is reproducible.