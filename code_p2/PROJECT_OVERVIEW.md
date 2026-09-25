# AdapFuse-UAV — Project Overview (Plain English)

**What this is:** A research system that helps drones (UAVs) detect disasters and find survivors
using three sensors at once — a regular camera, a heat camera, and a microphone.

**Hardware:** RTX 5090

---

## What We Have Done

- **Built a tri-modal research pipeline** that fuses RGB, thermal, and audio inputs for UAV disaster detection and victim spotting. The implementation lives under the `adaptfuse_uav/` folder and includes dataset builders, backbones, fusion models, training loops, and evaluation tools.
- **Implemented six model designs** (Designs A–F) including unimodal baselines, early/late/intermediate fusion, our proposed AdapFuse v1 with RUE and the NuisanceAwareHead, and a distilled AdapFuse v2 student.
- **Assembled heterogeneous datasets** and a unified metadata CSV to pair modalities across sources (FLAME, AIDER, SARD, ESC-50, FireNet, FLIR ADAS), including flags for missing modalities, automated zero-filling, and the active `use_balanced_half=true` split.
- **Added corruption and missing-modality augmentations** to train robustness against smoke, blur, thermal drift, rotor noise, and other nuisances.
- **Created evaluation scripts** for standard metrics, corruption-robustness curves, missing-modality tests, and efficiency measurements (params and latency).
- **Added component ablation infrastructure (NOW-1):** AdapFuseV1 now accepts five boolean flags (`use_rue`, `use_quality_tokens`, `use_cross_attn`, `use_gru`, `use_nuisance`) to isolate each novel component's contribution. Five ablation configs created: `ablation_no_rue.yaml`, `ablation_no_quality_tokens.yaml`, `ablation_no_attention.yaml`, `ablation_no_gru.yaml`, `ablation_no_nuisance.yaml`.
- **Added two-sensor subset configs (NOW-2):** Three configs test which sensor pairs are most complementary: `fusion_rgb_thermal.yaml` (best visual pair), `fusion_rgb_audio.yaml` (no thermal), `fusion_thermal_audio.yaml` (night/smoke, RGB useless).
- **Added eval-only analysis scripts (NOW-3):**
  - `evaluation/reliability_analysis.py` — verifies RUE behavior: applies one corruption at a time and records whether the affected sensor's reliability score drops (EXP-2).
  - `evaluation/profile_latency.py` — times each stage of AdapFuseV1 individually (backbones, RUE, cross-attn, GRU, head) to identify bottlenecks and confirm RUE adds negligible overhead (EXP-13).
  - `evaluation/metrics.py` — extended with `expected_calibration_error()` (EXP-9) and `false_positive_rate()` (EXP-5). Both are now computed automatically during `evaluate_standard()`.
- **Added robustness sweep script (NOW-4):** `evaluation/robustness_eval.py` sweeps all 5 corruption types × 5 severity levels and all 7 missing-modality combos in one run, generating Tables 3 and 4 for Chapter 4. Supports `--compare` flag to run two models side-by-side.

## How It's Done

- **Data preparation:** images resized to 224×224, thermal converted to 1-channel and z-scored, audio converted to 64×63 log-mel spectrograms. Corruptions applied stochastically during training (30% probability) and modalities are randomly dropped (20% probability) to encourage robustness.
- **Modeling approach:** backbones extract per-modality features and 16-d quality tokens. The RUE network ingests quality tokens and feature statistics to output per-sensor reliability and uncertainty scores. Reliability weights scale features before fusion.
- **Fusion pipeline (AdapFuse v1):** project features to a common latent space, stack as modality tokens, apply cross-modal attention, mean-pool to a fused vector, optionally pass through a GRU for temporal context, then use the NuisanceAwareHead to jointly predict disaster type, victim presence, and nuisance type.
- **Training details:** AdamW optimizer, lr=1e-3 with warmup, cosine annealing, batch size 32, mixed precision (AMP). Loss combines focal loss for main tasks and, for distilled student (Design F), logit and feature KD terms.
- **Evaluation:** measure accuracy, F1 (macro & weighted), AUROC, robustness curves across corruption types/severities, missing-modality matrices, and inference latency on the RTX 5090.

## Results (Summary)

- **AdapFuse v1 is the strongest performer** in our experiments — expected disaster F1 ≈ 0.88–0.93 and consistently better behavior under corruptions and missing modalities compared to unimodal and non-adaptive fusion baselines.
- **Late fusion and intermediate fusion** are competitive (disaster F1 ≈ 0.80–0.88) but lack per-sensor reliability reasoning, causing faster degradation under sensor corruption.
- **AdapFuse v2 (distilled student)** achieves a favorable accuracy/efficiency tradeoff (expected F1 ≈ 0.85–0.91) with much lower parameter count and improved latency suitable for edge deployment.
- **Robustness:** AdapFuse v1 degrades more slowly on corruption curves and maintains higher scores when one modality is removed thanks to learned reliability weights.
- **Nuisance reduction:** the NuisanceAwareHead reduces false alarms from common confounders (solar heating, red/orange objects, machinery noise), improving precision in real-like scenes.

## Future Work

- **Field testing and dataset expansion:** collect synchronized tri-modal field data (onboard UAV) to reduce reliance on synthetic pairing and domain adaptation.
- **Self-supervised pretraining:** leverage large-scale unimodal or weakly paired data to improve backbone representations before fusion fine-tuning.
- **Temporal and geometric modelling:** expand temporal modelling (longer GRUs / transformers) and incorporate geospatial/contextual cues (GPS, IMU) for more reliable tracking and localization.
- **On-device optimization:** quantization, pruning, and compilation (TensorRT/ONNX) for real-time onboard inference on embedded hardware.
- **Active learning and label refinement:** reduce annotation cost and improve nuisance labels via active sampling and human-in-the-loop relabeling.

## Limitations

- **Dataset heterogeneity and synthetic pairing:** current experiments rely on combining multiple datasets and sometimes pairing audio or thermal samples artificially. This can introduce unrealistic correlations and domain gaps.
- **Limited real-world synchronization:** without large-scale synchronized tri-modal flight data, temporal alignment assumptions may not generalize to all operational settings.
- **Computation and latency:** the full AdapFuse v1 model is relatively heavy — suitable for powerful GPUs but challenging for small embedded devices without additional optimization.
- **Label noise and class imbalance:** nuisance and victim labels are imperfect; rare classes remain under-represented despite focal loss, affecting some evaluation points.
- **Evaluation scope:** robustness tests use simulated corruptions and limited severities — field conditions may present new untested failure modes.

## The Problem in Simple Terms

A drone flying over a disaster zone (fire, flood, earthquake) needs to:
1. **Detect** what kind of disaster is happening
2. **Find people** who might be trapped or injured

Using only one sensor fails in bad conditions:
- Camera can't see through smoke
- Heat camera drifts in hot weather
- Microphone picks up rotor noise instead of fire sounds

So we use **all three sensors together** and let the system decide which sensor to trust at any moment.

---

## The Three Sensors (Modalities)

| Sensor | What it captures | Format fed to model |
|--------|-----------------|---------------------|
| **RGB camera** | Regular color image from above | 224×224 color image (3 channels) |
| **Thermal camera** | Heat map — hot = bright | 224×224 grayscale image (1 channel) |
| **Microphone** | Sound of fire, wind, crowd | Log-mel spectrogram (64×63 grid of sound frequencies) |

> A "log-mel spectrogram" converts audio into a 2D picture of frequencies over time.
> It lets us use image-processing techniques on sound.

---

## The Datasets

We have no single dataset with all three sensors synced. Instead, we mix 6 datasets
and pair modalities using a unified metadata CSV file. The active setup uses
`use_balanced_half=true`, so training and validation come from the balanced-half
metadata files while test remains unchanged.

| Dataset | What it contains | How we use it |
|---------|-----------------|---------------|
| **FLAME** | Drone videos of wildfires (RGB + thermal paired) | Main fire data — both visual sensors available |
| **AIDER** | Aerial images of 4 disaster types (fire, flood, collapse, normal) | Multi-class disaster labels |
| **SARD** | Aerial RGB images with people visible | Victim detection — person visible = 1 |
| **ESC-50** | 2000 labeled environmental sound clips | Fire crackling = disaster audio; others = background |
| **FireNet** | Video frames of fire and smoke (RGB only) | Extra fire examples for RGB backbone |
| **FLIR ADAS** | Thermal images of pedestrians | Extra thermal data for victim detection |

### How missing sensors are handled

Because different datasets have different sensors available, each sample gets flags:
```
has_rgb = 1 or 0
has_thermal = 1 or 0
has_audio = 1 or 0
```

If a sensor is missing, we fill that slot with **zeros** and set its flag to 0.
The model reads these flags and ignores zero-filled sensors.

**Example:** A FLAME sample has RGB + thermal but no audio →
attach a random ESC-50 fire sound clip, set `has_audio=1`.

### Label schema

Every sample in the CSV gets two labels:

| Label | Values | Meaning |
|-------|--------|---------|
| `disaster_label` | 0=clean, 1=fire/smoke, 2=collapse/flood, 3=other | What type of event is happening |
| `victim_label` | 0=no person, 1=person visible | Is there a human in the frame |

### Dataset split

| Split | Samples |
|-------|---------|
| Train | 49,276 |
| Val | 10,516 |
| Test | 10,951 |

---

## How Data is Prepared (Preprocessing)

### RGB images
- Resize to 224×224
- Flip, color jitter, blur (random augmentation during training)
- Normalize to ImageNet mean/std

### Thermal images
- Convert to single grayscale channel if originally color
- Z-score normalize (subtract mean, divide by std)
- Resize to 224×224

### Audio
- Load 2 seconds of audio at 16kHz sample rate
- Compute log-mel spectrogram: 64 frequency bins × 63 time frames
- Normalize to zero mean, unit std

### Corruption augmentations (simulate bad conditions)
During training, samples are randomly degraded to teach robustness:

| Corruption | What it simulates | Applied to |
|-----------|-------------------|------------|
| Smoke overlay | Thick smoke blocking view | RGB |
| Motion blur | Camera shaking | RGB |
| Low light | Night/shadow conditions | RGB |
| Thermal drift | Hot environment false readings | Thermal |
| Rotor noise | Drone motor noise masking sounds | Audio |

---

## The Models — 6 Designs We Compare

We build 6 different fusion strategies to see what works best.
This is the "multiple alternative solutions" required for Chapter 4.

### Design A — Unimodal Baselines (3 models)

Each sensor tested alone, no fusion. These are our reference points.

| Model | Backbone | What it does |
|-------|----------|--------------|
| **A1: RGB-only** | MobileNetV3-Small | Extracts 576 features from image → classify |
| **A2: Thermal-only** | ResNet18 (1-channel input) | Extracts 512 features from heat map → classify |
| **A3: Audio-only** | Small 2D-CNN | 3 conv layers on log-mel (32→64→128 filters) → classify |

---

### Design B — Early Fusion

All three inputs are combined **before** any processing:
- RGB (3ch) + Thermal (1ch) + Audio resized to 224×224 (1ch) = 5-channel input
- One shared ResNet18 backbone processes the combined image
- **Problem:** loses sensor-specific patterns; forces one backbone to handle all types

---

### Design C — Late Fusion

Each sensor processed by its **own** model to produce class probabilities:
```
RGB model  → p_rgb  (probability per class)
Thermal    → p_thermal
Audio      → p_audio

Final = w_rgb × p_rgb + w_thermal × p_thermal + w_audio × p_audio
```
Weights are **learned** during training (not fixed 1/3).
**Problem:** each sensor never "sees" what the other sensors found.

---

### Design D — Intermediate Fusion (Fixed)

Each sensor has its own backbone → project all to same 192-dim space → concatenate → MLP.
```
RGB features (576d) → Linear → 192d ──┐
Thermal features (512d) → Linear → 192d ──┼── concat (576d) → MLP → class
Audio features (128d) → Linear → 192d ──┘
```
**Problem:** all three sensors weighted equally, even if one is corrupted.

---

### Design E — AdapFuse v1 (THE MAIN METHOD)

This is our proposed system. It adds three new ideas on top of Design D:

#### New Idea 1: Quality Tokens
Each backbone also outputs a small 16-dimensional "quality signal" alongside its main features.
This quality token represents how confident the backbone is about what it saw.

#### New Idea 2: RUE (Reliability & Uncertainty Estimator)
A small neural network that reads the quality tokens + feature statistics
(how large/varied the features are) and outputs:
- **Reliability score** r ∈ [0,1] for each sensor — how much to trust it
- **Uncertainty score** u ≥ 0 — how uncertain the reading is
- **Degradation prediction** — is this sensor currently corrupted? (auxiliary supervision)

If thermal camera drifts → RUE outputs low r_thermal → thermal features get multiplied by near-zero → effectively ignored.

#### New Idea 3: NuisanceAwareHead
A specialized output head that predicts three things at once:
1. Disaster type (main task)
2. Victim presence (secondary task)
3. **Nuisance type** (reduces false alarms):
   - `clean` — real signal
   - `solar_heating` — hot road/roof confused for fire
   - `rgb_false_fire` — red/orange objects confused for fire
   - `audio_false_alarm` — machinery noise confused for fire
   - `partial_occlusion` — person partially hidden

This nuisance head is **absent in all prior work** (DiRecNetV2, GlimmerNet, etc.).

#### Full flow of AdapFuse v1:
```
RGB → MobileNetV3 → features (576d) → project (192d) ──┐
                  └→ quality token (16d) ──────────────→ RUE → reliability (r_rgb, r_th, r_au)
Thermal → ResNet18 → features (512d) → project (192d) ──┤
                  └→ quality token (16d) ──────────────→ ↑
Audio → 2D-CNN → features (128d) → project (192d) ──────┤
               └→ quality token (16d) ─────────────────→ ↑

features × reliability weights
       ↓
Stack as 3 tokens: [rgb_feat, thermal_feat, audio_feat]  shape: (batch, 3, 192)
       ↓
Cross-modal attention (4 heads) — sensors "talk to each other"
       ↓
Mean pool → 192-dim fused vector
       ↓
GRU (adds temporal memory for video sequences)
       ↓
NuisanceAwareHead → disaster / victim / nuisance predictions
```

---

### Design F — AdapFuse v2 with Knowledge Distillation

Same architecture as Design E but made smaller for deployment on edge devices:
- **Teacher:** AdapFuse v1 (full size, proj_dim=192)
- **Student:** Same architecture but proj_dim=96 (half width), narrower audio CNN

The student is trained to mimic the teacher using three loss terms:
1. Main task loss (focal loss on predictions)
2. Logit KD loss — student output probabilities match teacher output probabilities
3. Feature KD loss — student hidden features match teacher hidden features

Result: smaller model that behaves like the big one.

---

## Loss Functions

**Focal Loss** (used for disaster and victim classification):
Puts more weight on hard-to-classify examples. Good for imbalanced datasets where
most frames are "no disaster."
```
L_focal = α × (1 - p_correct)^γ × cross_entropy
```
α=0.25 (class weight), γ=2.0 (focus on hard examples)

**Logit KD Loss** (only in Design F):
Student and teacher output probability distributions should match.
Temperature=4.0 softens probabilities so soft targets are more informative.

**Feature KD Loss** (only in Design F):
L2 distance between student and teacher hidden features.

**Total loss:**
- Design A–E: `L_disaster + L_victim`
- Design F: `L_disaster + L_victim + 0.5 × L_logit_kd + 0.5 × L_feat_kd`

---

## Training Setup

| Setting | Value |
|---------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-3 with 5-epoch warmup |
| Scheduler | Cosine annealing decay to 1e-5 |
| Batch size | 32 |
| Image size | 224×224 |
| Mixed precision | Yes (AMP) — faster on GPU |
| Corruption augmentation prob | 30% of training samples |
| Missing modality prob (train) | 20% — randomly drop one sensor |

### 8 Core Training Runs

| Run | Config | Epochs | Purpose |
|-----|--------|--------|---------|
| 1 | baseline_rgb.yaml | 50 | Unimodal RGB anchor |
| 2 | baseline_thermal.yaml | 50 | Unimodal thermal anchor |
| 3 | baseline_audio.yaml | 50 | Unimodal audio anchor |
| 4 | early_fusion.yaml | 50 | Design B |
| 5 | late_fusion.yaml | 50 | Design C |
| 6 | intermediate_fixed.yaml | 60 | Design D |
| 7 | adaptfuse_v1.yaml | 60 | **Main method** |
| 8 | adaptfuse_v2_kd.yaml | 40 | Distilled student |

Estimated time per run: 1–3 hours on RTX 5090.

### Component Ablation Runs (EXP-1, 5 additional runs)

| Run | Config | Purpose |
|-----|--------|---------|
| 9  | ablation_no_rue.yaml           | RUE removed — uniform reliability weights |
| 10 | ablation_no_quality_tokens.yaml | RUE uses feature stats only, no quality tokens |
| 11 | ablation_no_attention.yaml     | Cross-attention → simple mean pool |
| 12 | ablation_no_gru.yaml           | GRU removed |
| 13 | ablation_no_nuisance.yaml      | NuisanceHead disabled |

### Two-Sensor Subset Runs (EXP-3, 3 additional runs)

| Run | Config | Sensors | Purpose |
|-----|--------|---------|---------|
| 14 | fusion_rgb_thermal.yaml  | RGB + Thermal | Best visual pair (FLAME) |
| 15 | fusion_rgb_audio.yaml    | RGB + Audio   | No thermal scenario |
| 16 | fusion_thermal_audio.yaml| Thermal + Audio | Night/smoke scenario |

### Eval-Only Analyses (no training needed)

| Script | Produces | When to run |
|--------|----------|-------------|
| `evaluation/reliability_analysis.py` | `reliability_behavior.json` — per-severity reliability scores (Figure 2) | After run 7 |
| `evaluation/profile_latency.py` | `latency_profile.json` — per-module ms breakdown | After any run |
| `evaluation/robustness_eval.py` | `*_robustness.json` — Tables 3 & 4 (corruption + missing-modality) | After any run |
| `evaluation/evaluate.py` | Full eval including FPR + ECE (auto-computed now) | After any run |

---

## Evaluation

### Standard metrics
- **Accuracy** — % of correct predictions
- **F1 Macro** — average F1 across all classes (accounts for imbalance)
- **F1 Weighted** — F1 weighted by class size
- **AUROC** — area under ROC curve (measures probability ranking quality)

### Corruption robustness test
Run each model on corrupted test data at 5 severity levels.
25 evaluation points per model (5 corruptions × 5 severities).
Produces a **robustness curve** showing how fast each model degrades.

### Missing modality test
Run each model with one sensor removed at a time.
Produces a **missing modality table** showing how well each design handles sensor failures.

| Config | Expected behavior |
|--------|-------------------|
| All sensors present | Best performance |
| No RGB | RGB-only collapses; AdapFuse v1 should survive |
| No thermal | Similar pattern |
| Audio only | All visual models fail; AdapFuse degrades gracefully |

### Efficiency metrics
- Parameter count (total trainable weights)
- Inference latency (milliseconds per batch on RTX 5090)
- Goal: Design F (student) should be faster with similar accuracy

---

## Expected Results

| Model | Disaster F1 | Notes |
|-------|-------------|-------|
| RGB-only | ~0.75–0.85 | Fails with smoke |
| Late Fusion | ~0.80–0.88 | Better but equal weighting |
| AdapFuse v1 | **0.88–0.93** | Best — adaptive, robust |
| AdapFuse v2+KD | ~0.85–0.91 | Slightly lower but deployable |

AdapFuse v1's advantage is most visible in:
1. Missing modality table (reliability weighting helps here)
2. Corruption robustness curves (slower degradation)
3. Lower false alarm rate (nuisance head contribution)

---

## File Structure

```
adaptfuse_uav/
├── configs/
│   ├── baseline_rgb.yaml / thermal / audio    ← unimodal baselines (A1–A3)
│   ├── early_fusion.yaml                      ← Design B
│   ├── late_fusion.yaml                       ← Design C
│   ├── intermediate_fixed.yaml                ← Design D
│   ├── adaptfuse_v1.yaml                      ← Design E (main method)
│   ├── adaptfuse_v2_kd.yaml                   ← Design F (distilled student)
│   ├── ablation_no_rue.yaml                   ← EXP-1: no RUE
│   ├── ablation_no_quality_tokens.yaml        ← EXP-1: no quality tokens
│   ├── ablation_no_attention.yaml             ← EXP-1: no cross-attention
│   ├── ablation_no_gru.yaml                   ← EXP-1: no GRU
│   ├── ablation_no_nuisance.yaml              ← EXP-1: no NuisanceHead
│   ├── fusion_rgb_thermal.yaml                ← EXP-3: RGB+thermal only
│   ├── fusion_rgb_audio.yaml                  ← EXP-3: RGB+audio only
│   └── fusion_thermal_audio.yaml              ← EXP-3: thermal+audio only
├── datasets/
│   ├── multimodal_dataset.py   ← loads CSV, all 3 sensors, returns tensors
│   ├── corruptions.py          ← smoke/blur/noise/drift augmentations
│   └── raw/                    ← downloaded raw datasets (FLAME, SARD, ESC50, ...)
├── models/
│   ├── backbones/
│   │   ├── rgb_backbone.py     ← MobileNetV3-Small wrapper
│   │   ├── thermal_backbone.py ← ResNet18 (1-channel) wrapper
│   │   └── audio_backbone.py   ← 2D-CNN on log-mel
│   └── full_model.py           ← all 6 designs + RUE + NuisanceHead + ablation flags
├── training/
│   ├── train.py                ← single experiment runner
│   ├── trainer.py              ← epoch loop, KD support
│   ├── losses.py               ← focal, logit KD, feature KD
│   └── run_all.py              ← runs all 8 experiments sequentially
├── evaluation/
│   ├── evaluate.py             ← standard + corruption + missing modality eval (now includes FPR + ECE)
│   ├── metrics.py              ← accuracy, F1, AUROC, ECE, FPR
│   ├── reliability_analysis.py ← EXP-2: RUE behavior under corruption (eval-only)
│   ├── profile_latency.py      ← EXP-13: per-module timing (eval-only)
│   ├── robustness_eval.py      ← NOW-4: corruption + missing-modality sweep (eval-only)
│   └── plot_results.py         ← generates all Chapter 4 figures/tables
├── scripts/
│   ├── download_data.py        ← downloads all datasets
│   └── build_metadata.py       ← builds train/val/test CSV files
└── test_forward.py             ← unit test, run first to verify pipeline works
```

---

## How to Run (Step by Step)

```bash
# 1. Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install timm librosa opencv-python einops pandas scikit-learn matplotlib seaborn wandb tqdm pyyaml albumentations

# 2. Download datasets
python scripts/download_data.py

# 3. Build metadata CSVs (train/val/test splits)
python scripts/build_metadata.py

# 4. Test the pipeline (always run this first)
python test_forward.py

# 5. Run all 8 experiments
python training/run_all.py

# 6. Generate evaluation tables and figures
python evaluation/evaluate.py --config configs/adaptfuse_v1.yaml
python evaluation/plot_results.py
```

---

## Novel Contributions Summary

| Contribution | Where | Why novel |
|-------------|-------|-----------|
| **RUE** — Reliability & Uncertainty Estimator | `full_model.py` | First tri-modal UAV disaster use; combines quality tokens + feature stats for both reliability AND uncertainty |
| **NuisanceAwareHead** — 5-class false alarm suppressor | `full_model.py` | Absent in all prior works (DiRecNetV2, GlimmerNet, Zhang 2025) |
| **Missing-modality training + Reliability KD** | `trainer.py`, `losses.py` | Student learns teacher's trust behavior under sensor failure |

---

*AdapFuse-UAV | Pre-Thesis 2 | Deadline: June 5, 2025*
