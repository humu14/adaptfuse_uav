# Pre-Thesis 2 — Full Experiment Implementation Plan
## AdapFuse-UAV: Reliability-Aware Multimodal Fusion for UAV Disaster Detection

**Deadline:** June 5, 2025  
**Hardware:** RTX 9060 (local), BRAC CSE Lab access if needed  
**Target:** Chapter 4 (Design Process + Preliminary Design + Simulation/Functional Verification)  
**Rubric Goals:** Design multiple alternative solutions, simulate them for functional verification

---

## 0. Reality Check — What is Achievable by June 5

Given your hardware and deadline, here is what is realistic:

| What the full plan says | What you implement for Pre-Thesis 2 |
|---|---|
| SAME + TARE + RRAE custom backbones | MobileNetV3-Small (RGB) + ResNet18-1ch (Thermal) + 2D-CNN (Audio) with light modifications |
| Full AAM + RUE + TFM fusion | Reliability-weighted intermediate fusion with simple quality estimator |
| Teacher + student KD pipeline | Teacher only trained; student used in ablation |
| Full corruption curriculum | Core degradation augmentations (smoke, blur, noise) |
| Jetson/TensorRT deployment | Parameter count + latency measured on RTX9060, projected to edge |
| 7 KD loss terms | 3 loss terms: focal task + logit KD + feature KD |

This is still more than enough for a strong Chapter 4 with real simulation results.

---

## 1. Repository Setup (Day 1)

### 1.1 Folder structure

```
adaptfuse_uav/
├── configs/
│   ├── baseline_rgb.yaml
│   ├── baseline_thermal.yaml
│   ├── baseline_audio.yaml
│   ├── early_fusion.yaml
│   ├── late_fusion.yaml
│   ├── adaptfuse_v1.yaml
│   └── adaptfuse_v2_kd.yaml
├── data/
│   └── metadata/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
├── datasets/
│   ├── raw/           ← downloaded datasets go here
│   │   ├── FLAME/
│   │   ├── AIDER/
│   │   ├── ESC50/
│   │   └── SARD/
│   ├── processed/     ← after preprocessing
│   └── splits/
├── models/
│   ├── backbones/
│   │   ├── rgb_backbone.py
│   │   ├── thermal_backbone.py
│   │   └── audio_backbone.py
│   ├── fusion/
│   │   ├── reliability_estimator.py
│   │   ├── fusion_module.py
│   │   └── temporal_module.py
│   ├── heads/
│   │   ├── disaster_head.py
│   │   └── victim_head.py
│   └── full_model.py
├── training/
│   ├── train.py
│   ├── losses.py
│   └── trainer.py
├── evaluation/
│   ├── evaluate.py
│   ├── metrics.py
│   └── corruption_eval.py
├── scripts/
│   ├── download_data.sh
│   ├── preprocess_flame.py
│   ├── preprocess_aider.py
│   └── build_metadata.py
├── notebooks/
│   └── EDA.ipynb
└── outputs/
    ├── checkpoints/
    ├── logs/
    └── figures/
```

### 1.2 Environment setup

```bash
conda create -n adaptfuse python=3.10
conda activate adaptfuse

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install timm librosa opencv-python einops pandas scikit-learn matplotlib seaborn
pip install wandb tqdm pyyaml albumentations pillow scipy
```

### 1.3 Config system

Use plain YAML + argparse. No Hydra needed for thesis scale. Each config has:
```yaml
model: adaptfuse_v1
modalities: [rgb, thermal, audio]
num_classes_disaster: 3
num_classes_victim: 4
batch_size: 32
epochs: 50
lr: 1e-3
use_focal_loss: true
use_kd: false
kd_teacher_ckpt: null
corruption_prob: 0.3
missing_modality_prob: 0.2
```

---

## 2. Dataset Plan

### 2.1 Datasets you have — what to use from each

| Dataset | What you use | Label mapping |
|---|---|---|
| **FLAME** (RGB + thermal) | Core wildfire data — both modalities paired | disaster=1 (fire), victim=0 |
| **FireNet** (RGB video) | Fire/smoke detection frames | disaster=1 (fire/smoke), victim=0 |
| **AIDER** (RGB) | Multi-disaster: fire, flood, collapse, traffic | disaster=0/1/2/3, victim=0 |
| **ESC-50** (Audio) | crackling_fire, wind, rain, crowd classes | audio disaster/background label |
| **SARD** (RGB aerial) | Person detection aerial | victim=1, disaster varies |
| **FLIR ADAS** (Thermal) | Pedestrian thermal signatures | use as thermal victim source |

**Do NOT use MEDIC for now** — it is NLP-heavy and needs extra preprocessing for image extraction. Drop it for the June 5 deadline.

### 2.2 Unified label schema

Each sample in your metadata CSV gets:

```
sample_id, rgb_path, thermal_path, audio_path,
disaster_label,   # 0=clean, 1=fire/smoke, 2=collapse/flood, 3=other_disaster
victim_label,     # 0=no person, 1=person visible
has_rgb,          # 1 or 0
has_thermal,      # 1 or 0
has_audio,        # 1 or 0
source_dataset,   # flame / aider / sard / firenet / esc50 / flir
split,            # train / val / test
degradation,      # none / smoke / blur / noise / lowlight
```

### 2.3 How to handle missing modalities in the dataset

Because you do not have a single dataset with all three modalities synchronized, you pair modalities probabilistically:

- **FLAME samples** → has_rgb=1, has_thermal=1, has_audio=0 (pair with random ESC-50 fire audio)
- **AIDER samples** → has_rgb=1, has_thermal=0, has_audio=0
- **SARD samples** → has_rgb=1, has_thermal=0 or 1 (use FLIR pedestrian thermal if class matches), has_audio=0
- **ESC-50** → audio-only context, attached to matching visual samples

For missing modalities during training, the dataloader **zero-fills** the missing tensor and sets the `has_X` flag to 0. The model reads these flags.

### 2.4 Target dataset size

| Split | Samples |
|---|---|
| Train | ~5,000 |
| Val | ~1,000 |
| Test | ~1,000 |

This is achievable from the datasets you have. Do not oversample FLAME — mix all sources.

### 2.5 Build metadata script (scripts/build_metadata.py)

Key logic:
```python
# For each source, walk directories, build rows, assign labels
# FLAME: positive fire = 1, negative = 0
# AIDER: read subfolder names (fire, flood, collapsed_building, normal)
# SARD: all samples are victim=1, disaster context from filename
# ESC50: crackling_fire -> disaster_label=1, others -> 0
# Merge all into single CSV, shuffle, stratified split
```

---

## 3. Preprocessing Pipelines

### 3.1 RGB preprocessing

```python
# During training
transforms = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.3, contrast=0.3),
    A.GaussianBlur(p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Corruption augmentations (applied with prob corruption_prob)
def apply_smoke_overlay(img, severity=0.5):
    # add gray haze using alpha blend
    pass

def apply_low_light(img, gamma=2.5):
    pass

def apply_motion_blur(img, kernel_size=7):
    pass
```

### 3.2 Thermal preprocessing

```python
# Thermal is single channel (grayscale or pseudo-color)
# If pseudo-color RGB from FLAME: convert to grayscale first
# Normalize to [0,1], then z-score
def preprocess_thermal(img):
    if img.shape[2] == 3:  # pseudo-color
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    img = (img - img.mean()) / (img.std() + 1e-6)
    img = cv2.resize(img, (224, 224))
    return img[np.newaxis, ...]  # shape: (1, 224, 224)

# Corruption: add thermal drift (uniform offset), saturation noise
def apply_thermal_drift(img, offset=0.3):
    return np.clip(img + offset, -3, 3)
```

### 3.3 Audio preprocessing

```python
import librosa
import numpy as np

def load_audio_logmel(path, sr=16000, duration=2.0, n_mels=64, hop_length=512):
    y, _ = librosa.load(path, sr=sr, duration=duration)
    if len(y) < int(sr * duration):
        y = np.pad(y, (0, int(sr * duration) - len(y)))
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    # shape: (n_mels, time_frames) -> (1, 64, T)
    return log_mel[np.newaxis, ...]

# Corruption: mix rotor noise
def add_rotor_noise(log_mel, snr_db=5):
    # add synthetic rotor noise pattern (low-frequency noise burst)
    noise = np.random.randn(*log_mel.shape) * (10 ** (-snr_db / 20))
    return log_mel + noise
```

---

## 4. Model Architectures — Alternative Designs

This is the core of Chapter 4.2. You implement **4 alternative designs** plus a baseline, giving you the rubric-required "multiple alternative solutions."

### Design A — Unimodal Baselines (3 variants)

**A1: RGB-only**
```
MobileNetV3-Small -> GAP -> Linear(576, 256) -> ReLU -> Linear(256, num_classes)
```

**A2: Thermal-only**
```
ResNet18 (1-channel input) -> GAP -> Linear(512, 256) -> ReLU -> Linear(256, num_classes)
```

**A3: Audio-only**
```
2D-CNN on log-mel (3 layers: 32->64->128 filters) -> GAP -> Linear(128, 64) -> Linear(64, num_classes)
```

These are your ablation anchors.

---

### Design B — Early Fusion (Simple Concat)

All three modalities fused at input level:

```
RGB (224x224x3) -> resize + flatten
Thermal (224x224x1) -> resize + flatten  
Audio log-mel (64xT) -> resize to 224x224

Concatenate -> shared CNN backbone -> classification head
```

Problem: loses modality-specific representations. You demonstrate this limitation.

---

### Design C — Late Fusion (Decision-Level)

Three separate classifiers, outputs combined:

```
RGB  -> MobileNetV3 -> p_rgb  (disaster probs)
Thermal -> ResNet18 -> p_thermal
Audio -> AudioCNN -> p_audio

p_final = w_rgb * p_rgb + w_thermal * p_thermal + w_audio * p_audio
```

Where weights are fixed (1/3 each) in baseline, then learned in variant.

Problem: loses cross-modal information.

---

### Design D — Intermediate Fusion v1 (Fixed Weights)

The standard approach:

```python
class IntermediateFusionFixed(nn.Module):
    def __init__(self):
        self.rgb_enc = MobileNetV3Small()        # out: 576-d
        self.thermal_enc = ResNet18_1ch()         # out: 512-d
        self.audio_enc = AudioCNN()               # out: 128-d

        self.rgb_proj = nn.Linear(576, 192)
        self.thermal_proj = nn.Linear(512, 192)
        self.audio_proj = nn.Linear(128, 192)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(192*3, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, rgb, thermal, audio):
        f_rgb = self.rgb_proj(self.rgb_enc(rgb))
        f_th = self.thermal_proj(self.thermal_enc(thermal))
        f_au = self.audio_proj(self.audio_enc(audio))
        fused = torch.cat([f_rgb, f_th, f_au], dim=-1)
        return self.fusion_mlp(fused)
```

---

### Design E — AdapFuse v1: Reliability-Weighted Intermediate Fusion (YOUR PROPOSED METHOD)

This is your main contribution for Pre-Thesis 2.

```python
class ReliabilityEstimator(nn.Module):
    """Estimates per-modality reliability from feature quality signals."""
    def __init__(self, feat_dim=192, num_modalities=3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim * num_modalities, 64),
            nn.ReLU(),
            nn.Linear(64, num_modalities),
            nn.Sigmoid()   # outputs r_rgb, r_thermal, r_audio in [0,1]
        )

    def forward(self, f_rgb, f_th, f_au):
        combined = torch.cat([f_rgb, f_th, f_au], dim=-1)
        return self.mlp(combined)   # shape: (B, 3)


class AdapFuseV1(nn.Module):
    def __init__(self, num_disaster_classes=3, num_victim_classes=2):
        super().__init__()
        # Encoders
        self.rgb_enc = MobileNetV3Small()
        self.thermal_enc = ResNet18_1ch()
        self.audio_enc = AudioCNN()

        # Projectors to shared dim
        self.rgb_proj = nn.Linear(576, 192)
        self.thermal_proj = nn.Linear(512, 192)
        self.audio_proj = nn.Linear(128, 192)

        # Reliability estimator
        self.rue = ReliabilityEstimator(feat_dim=192, num_modalities=3)

        # Cross-modal attention (lightweight, 4 heads, 192-d)
        self.cross_attn = nn.MultiheadAttention(embed_dim=192, num_heads=4, batch_first=True)

        # GRU for temporal stability (over T=1 for static; enables future temporal use)
        self.gru = nn.GRU(input_size=192, hidden_size=192, batch_first=True)

        # Task heads
        self.disaster_head = nn.Linear(192, num_disaster_classes)
        self.victim_head = nn.Linear(192, num_victim_classes)

    def forward(self, rgb, thermal, audio, has_rgb, has_thermal, has_audio):
        # Encode
        f_rgb = self.rgb_proj(self.rgb_enc(rgb))      # (B, 192)
        f_th = self.thermal_proj(self.thermal_enc(thermal))
        f_au = self.audio_proj(self.audio_enc(audio))

        # Zero out missing modalities
        f_rgb = f_rgb * has_rgb.unsqueeze(-1)
        f_th = f_th * has_thermal.unsqueeze(-1)
        f_au = f_au * has_audio.unsqueeze(-1)

        # Estimate reliability
        reliability = self.rue(f_rgb, f_th, f_au)  # (B, 3)
        r_rgb, r_th, r_au = reliability[:, 0:1], reliability[:, 1:2], reliability[:, 2:3]

        # Weight features by reliability
        f_rgb = f_rgb * r_rgb
        f_th = f_th * r_th
        f_au = f_au * r_au

        # Stack as token sequence for attention: (B, 3, 192)
        tokens = torch.stack([f_rgb, f_th, f_au], dim=1)

        # Cross-modal attention
        attended, _ = self.cross_attn(tokens, tokens, tokens)

        # Mean pool across modalities
        fused = attended.mean(dim=1)   # (B, 192)

        # GRU (treat fused as T=1 sequence for now)
        fused, _ = self.gru(fused.unsqueeze(1))
        fused = fused.squeeze(1)

        # Heads
        disaster_out = self.disaster_head(fused)
        victim_out = self.victim_head(fused)

        return disaster_out, victim_out, reliability
```

---

### Design F — AdapFuse v2: + Knowledge Distillation (EXTENSION)

Same architecture as E but now the student is trained with:

```
L_total = L_focal_task + 0.5 * L_logit_kd + 0.5 * L_feature_kd
```

Teacher = AdapFuse v1 (full model)
Student = AdapFuse v1 with 0.5× channel width

This is your deployment-oriented variant.

---

## 5. Loss Functions

```python
import torch
import torch.nn.functional as F

def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """Focal loss for class imbalance."""
    ce = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce)
    loss = alpha * ((1 - pt) ** gamma) * ce
    return loss.mean()

def logit_kd_loss(student_logits, teacher_logits, temperature=4.0):
    """Knowledge distillation on output logits."""
    s = F.log_softmax(student_logits / temperature, dim=-1)
    t = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(s, t, reduction='batchmean') * (temperature ** 2)

def feature_kd_loss(student_feat, teacher_feat):
    """L2 feature-level distillation."""
    return F.mse_loss(student_feat, teacher_feat.detach())

def total_loss(disaster_out, victim_out, disaster_gt, victim_gt,
               teacher_disaster=None, teacher_victim=None,
               student_feat=None, teacher_feat=None,
               use_kd=False, lambda_kd=0.5):
    
    l_disaster = focal_loss(disaster_out, disaster_gt)
    l_victim = focal_loss(victim_out, victim_gt)
    l_task = l_disaster + l_victim

    if use_kd and teacher_disaster is not None:
        l_logit = logit_kd_loss(disaster_out, teacher_disaster)
        l_feat = feature_kd_loss(student_feat, teacher_feat)
        return l_task + lambda_kd * l_logit + lambda_kd * l_feat
    
    return l_task
```

---

## 6. Training Pipeline

### 6.1 Trainer (training/trainer.py)

```python
class Trainer:
    def __init__(self, model, optimizer, scheduler, config, teacher=None):
        self.model = model
        self.teacher = teacher  # None for baseline runs
        ...

    def train_epoch(self, loader):
        for batch in loader:
            rgb, thermal, audio = batch['rgb'], batch['thermal'], batch['audio']
            has_rgb, has_thermal, has_audio = batch['has_rgb'], ...
            disaster_gt, victim_gt = batch['disaster_label'], batch['victim_label']

            disaster_out, victim_out, reliability = self.model(
                rgb, thermal, audio, has_rgb, has_thermal, has_audio
            )
            
            # Get teacher outputs if using KD
            if self.teacher is not None:
                with torch.no_grad():
                    t_disaster, t_victim, _ = self.teacher(rgb, thermal, audio, ...)
            
            loss = total_loss(disaster_out, victim_out, disaster_gt, victim_gt,
                              teacher_disaster=t_disaster if self.teacher else None,
                              use_kd=self.teacher is not None)
            
            loss.backward()
            self.optimizer.step()
```

### 6.2 Training schedule

| Run | Config | Epochs | Purpose |
|---|---|---|---|
| Run 1 | baseline_rgb.yaml | 50 | Unimodal RGB baseline |
| Run 2 | baseline_thermal.yaml | 50 | Unimodal thermal baseline |
| Run 3 | baseline_audio.yaml | 50 | Unimodal audio baseline |
| Run 4 | early_fusion.yaml | 50 | Early fusion (Design B) |
| Run 5 | late_fusion.yaml | 50 | Late fusion (Design C) |
| Run 6 | intermediate_fixed.yaml | 60 | Intermediate fixed (Design D) |
| Run 7 | adaptfuse_v1.yaml | 60 | **AdapFuse v1 — main method** |
| Run 8 | adaptfuse_v2_kd.yaml | 40 | **AdapFuse v2 + KD** |

All runs on RTX 9060. Estimated time per run: 1–3 hours at batch_size=32, 224×224.

### 6.3 Optimizer settings

```yaml
optimizer: AdamW
lr: 1e-3
weight_decay: 1e-4
scheduler: CosineAnnealingLR
T_max: 50
min_lr: 1e-5
warmup_epochs: 5
```

---

## 7. Evaluation Plan

### 7.1 Standard metrics

```python
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def evaluate(model, loader):
    all_preds, all_targets = [], []
    all_probs = []
    
    for batch in loader:
        with torch.no_grad():
            disaster_out, victim_out, reliability = model(...)
        preds = disaster_out.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(batch['disaster_label'].cpu().numpy())
        all_probs.extend(F.softmax(disaster_out, dim=-1).cpu().numpy())
    
    return {
        'accuracy': accuracy_score(all_targets, all_preds),
        'f1_macro': f1_score(all_targets, all_preds, average='macro'),
        'f1_weighted': f1_score(all_targets, all_preds, average='weighted'),
        'auroc': roc_auc_score(all_targets, all_probs, multi_class='ovr')
    }
```

### 7.2 Corruption robustness evaluation

For the test set, run evaluation under 5 corruption levels each:

```python
corruptions = {
    'smoke_overlay': [0.1, 0.3, 0.5, 0.7, 0.9],       # severity 1-5
    'motion_blur': [3, 5, 7, 11, 15],                   # kernel size
    'low_light': [2.0, 2.5, 3.0, 3.5, 4.0],            # gamma
    'thermal_drift': [0.1, 0.3, 0.5, 0.7, 1.0],        # offset
    'rotor_noise_snr': [20, 15, 10, 5, 0],              # dB
}

# Run each corruption at each severity → 25 evaluation points per model
```

This produces your **robustness curve** — a key figure for Chapter 4.

### 7.3 Missing modality evaluation

```python
missing_configs = [
    {'has_rgb': 1, 'has_thermal': 1, 'has_audio': 1},   # full
    {'has_rgb': 0, 'has_thermal': 1, 'has_audio': 1},   # no RGB
    {'has_rgb': 1, 'has_thermal': 0, 'has_audio': 1},   # no thermal
    {'has_rgb': 1, 'has_thermal': 1, 'has_audio': 0},   # no audio
    {'has_rgb': 0, 'has_thermal': 0, 'has_audio': 1},   # audio only
    {'has_rgb': 1, 'has_thermal': 0, 'has_audio': 0},   # RGB only
]

# For each config, evaluate all models on test set
```

This produces your **missing-modality robustness table**.

### 7.4 Latency and efficiency

```python
def measure_latency(model, input_dict, n_runs=100, device='cuda'):
    model.eval()
    model.to(device)
    
    # Warmup
    for _ in range(10):
        model(**input_dict)
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_runs):
        with torch.no_grad():
            model(**input_dict)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) / n_runs * 1000  # ms
    return elapsed

# Also count parameters
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```

---

## 8. Chapter 4 Content Mapping

Here is exactly what goes in the report.

### Chapter 4.1 — Design Process / Methodology Overview

Write this section in the order:

1. **Design requirements** — what constraints the system must satisfy (latency <500ms, handles missing modalities, handles smoke/noise)
2. **Design space** — what design dimensions you explored (backbone choice, fusion strategy, reliability estimation, loss design)
3. **Evaluation criteria** — what you used to choose between alternatives (F1, AUROC, robustness under corruption, parameter count)
4. **Alternative generation process** — Designs A through F and why each one adds something

This is ~1,500 words with 1 system-design flowchart.

**Figure to include:**  
A flowchart: Dataset → Preprocessing → Design A / B / C / D / E → Evaluation → Best Design

### Chapter 4.2 — Preliminary Design / Design Specification

For each design alternative, write:

| Field | What to fill |
|---|---|
| Design name | A1 RGB-Only / B Early Fusion / D IntermFixed / E AdapFuse-v1 |
| Architecture | Short description + module diagram |
| Key hypothesis | Why this design should work |
| Parameters | Count |
| Expected tradeoff | Speed vs accuracy vs robustness |

Then write the detailed architecture of **Design E (AdapFuse v1)** as your proposed design. Include:
- Full model diagram
- RUE block diagram
- Loss formula
- Training setup

### Chapter 4 — Simulation / Functional Verification

Present your experiment results as tables and figures:

**Table 1 — Baseline Comparison**

| Model | Accuracy | F1 Macro | AUROC | Params | Latency (ms) |
|---|---|---|---|---|---|
| RGB-only (A1) | | | | | |
| Thermal-only (A2) | | | | | |
| Audio-only (A3) | | | | | |
| Early Fusion (B) | | | | | |
| Late Fusion (C) | | | | | |
| Intermediate Fixed (D) | | | | | |
| **AdapFuse v1 (E)** | | | | | |
| AdapFuse v2 + KD (F) | | | | | |

**Table 2 — Missing Modality Robustness**

| Missing Config | RGB-only | Late Fusion | AdapFuse v1 |
|---|---|---|---|
| Full (all present) | | | |
| No RGB | — | | |
| No Thermal | | | |
| No Audio | | | |

**Table 3 — Corruption Robustness (F1 at Severity 3)**

| Corruption | RGB-only | Intermediate Fixed | AdapFuse v1 |
|---|---|---|---|
| Smoke overlay | | | |
| Motion blur | | | |
| Thermal drift | | | |
| Rotor noise | | | |

**Figure 1 — Robustness curves:** F1 vs corruption severity for all key models

**Figure 2 — Reliability score behavior:** Scatter or bar showing r_rgb, r_thermal, r_audio changing across corruption types

**Figure 3 — Architecture diagram:** Full AdapFuse v1 pipeline

---

## 9. Poster Content Plan (for Presentation Marks)

The poster should cover:

1. **Problem** — UAV disaster detection, why single-modal fails (2 panels)
2. **Proposed System** — AdapFuse v1 architecture diagram (1 large panel)
3. **Datasets used** — FLAME, AIDER, SARD, ESC-50 (1 panel)
4. **Key results** — Table 1 (design comparison), Table 2 (missing modality) (2 panels)
5. **Key insight** — Reliability score visualization (1 panel)
6. **Conclusion & future work** (1 panel)

---

## 10. Week-by-Week Schedule to June 5

| Days | Tasks |
|---|---|
| **Day 1–2** | Repo setup, env install, download all datasets, verify loading |
| **Day 3–4** | Write metadata CSVs (train/val/test splits), write all three modality loaders |
| **Day 5** | Preprocessing pipeline + corruption functions + missing-modality wrapper |
| **Day 6** | EDA notebook: visualize 30 samples, confirm class balance |
| **Day 7–8** | Implement and train Designs A1, A2, A3 (unimodal baselines) |
| **Day 9** | Implement and train Designs B, C (early + late fusion) |
| **Day 10** | Implement and train Design D (intermediate fixed) |
| **Day 11–12** | Implement and train Design E (AdapFuse v1 — full method) |
| **Day 13** | Implement and train Design F (AdapFuse v2 + KD) |
| **Day 14** | Run all corruption evaluations + missing modality evaluations |
| **Day 15** | Generate all tables and figures |
| **Day 16–17** | Write Chapter 4.1 and 4.2 |
| **Day 18** | Finalize full report, prepare poster |
| **Day 19 (buffer)** | Final review, fix any failing experiments |
| **Day 20 (June 5)** | Submit |

---

## 11. What to Write When Results Are Not Perfect

For thesis purposes, imperfect results that are well-analyzed are better than cherry-picked perfect ones.

If AdapFuse v1 does not beat all baselines on every metric:

- Show it beats baselines specifically on the missing-modality table (this is where reliability-weighting should clearly help)
- Show it degrades more gracefully on the corruption robustness curve
- Discuss failure cases honestly — this is what Chapter 4 analysis is for

The rubric rewards design process, alternatives, and simulation — not a perfect number.

---

## 12. Minimum Viable Experiment (If Time Runs Out)

If you only have 5 days left before June 5, run these and nothing else:

1. RGB-only baseline → 1 number
2. Thermal-only baseline → 1 number
3. Late fusion (Design C) → 1 number
4. AdapFuse v1 (Design E) → main result
5. Missing-modality table for these 4 models
6. One corruption curve (smoke only)

That is enough for Tables 1, 2, and Figure 1. You can describe Designs B and D without training them if you have architecture diagrams and parameter counts.

---

## 13. Key Files to Code First (Priority Order)

```
1. scripts/build_metadata.py            ← nothing works without data
2. datasets/multimodal_dataset.py       ← core loader
3. datasets/corruptions.py              ← augmentation transforms
4. models/backbones/rgb_backbone.py     ← simplest backbone first
5. models/backbones/thermal_backbone.py
6. models/backbones/audio_backbone.py
7. models/full_model.py (Design A, then D, then E)
8. training/losses.py
9. training/train.py
10. evaluation/evaluate.py
```

Write in this exact order. Do not touch the distillation code until all baselines are running.

---

*End of Pre-Thesis 2 Implementation Plan*  
*System name: AdapFuse-UAV | Target: Chapter 4 Design + Simulation | Deadline: June 5*
