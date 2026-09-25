# AdapFuse-UAV — Reliability-Aware Multimodal Fusion for UAV Disaster Detection
## Pre-Thesis 2 | June 5 Deadline | RTX 9060

---

## Project Structure

```
adaptfuse_uav/
├── configs/               ← YAML configs for all 8 experiments
├── datasets/              ← Dataset loaders and corruption augmentations
├── models/backbones/      ← RGB, Thermal, Audio encoders
├── models/full_model.py   ← All 6 design alternatives (A1–F)
├── training/              ← Trainer, losses, train.py, run_all.py
├── evaluation/            ← Metrics, full evaluation, plotting
├── scripts/               ← Data download and metadata building
└── test_forward.py        ← Unit tests (run FIRST)
```

---

## Quick Start (Run in order)

### Step 1: Install dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Step 2: Download datasets & generate synthetic samples
```bash
python scripts/download_data.py
```
> Automatically downloads ESC-50, FLAME, AIDER, SARD, FireNet, and C2A when Kaggle credentials are configured.
> It also performs the dataset-specific restructuring needed by the metadata builder.

### Step 3: Build metadata CSVs
```bash
python scripts/build_metadata.py
```
This now also writes `train_balanced.csv` and `val_balanced.csv` after mixing in the downloaded datasets and oversampling the minority classes.

### Step 4: Verify the pipeline (ALWAYS RUN THIS FIRST)
```bash
python test_forward.py
```

### Step 5: Run all 8 experiments
```bash
# Run everything sequentially (8 experiments)
python training/run_all.py

# OR run individually:
python training/train.py --config configs/baseline_rgb.yaml
python training/train.py --config configs/adaptfuse_v1.yaml
```

### Step 6: Evaluate and generate figures
```bash
# Full evaluation (standard + corruption + missing modality)
python evaluation/evaluate.py --config configs/adaptfuse_v1.yaml

# Generate all Chapter 4 figures and tables
python evaluation/plot_results.py
```

---

## Model Designs (Chapter 4)

| Design | Model | Key Feature |
|--------|-------|-------------|
| A1 | RGB-Only | MobileNetV3-Small baseline |
| A2 | Thermal-Only | ResNet18 (1-channel) |
| A3 | Audio-Only | 2D-CNN on log-mel |
| B  | Early Fusion | Concat at input, shared CNN |
| C  | Late Fusion | Decision-level combination |
| D  | Intermediate Fixed | Feature-level concat, MLP |
| **E** | **AdapFuse v1** | **RUE + Cross-Attn + GRU + NuisanceHead** |
| F  | AdapFuse v2+KD | Distilled student (0.5× width) |

---

## Novel Contributions (from both plans)

1. **Reliability & Uncertainty Estimator (RUE)** — Per-modality reliability from quality tokens + feature statistics. First applied to tri-modal UAV disaster detection.

2. **NuisanceAwareHead** — Explicit 5-class nuisance classification to reduce false alarms (solar heating, RGB confusion, rotor noise). Absent in all prior works.

3. **Missing-modality training + Reliability KD** — Student learns teacher's trust behavior under sensor failure.

---

## Expected Results (Chapter 4 Tables)

| Model | Disaster F1 | Key Advantage |
|-------|------------|---------------|
| RGB-Only | ~0.75–0.85 | Baseline |
| Late Fusion | ~0.80–0.88 | Multi-sensor |
| AdapFuse v1 | **0.88–0.93** | Robust to degradation |
| AdapFuse v2+KD | ~0.85–0.91 | Edge-deployable |

---

## GPU Usage

All training uses:
- CUDA with AMP (mixed precision)
- `tqdm` progress bars for all loops
- Automatic GPU detection with CPU fallback

Check GPU: `python -c "import torch; print(torch.cuda.get_device_name(0))"`
