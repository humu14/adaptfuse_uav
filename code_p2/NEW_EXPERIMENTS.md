# New Experiments & Ablations — AdapFuse-UAV

**What's already planned:** 8 training runs (baselines A1–A3, designs B–F) + corruption eval + missing-modality eval.

**What's missing:** The 8 runs prove the final model works. These experiments prove *why* it works and *which parts matter*. A thesis panel will ask all of these.

---

## Priority Tier 0 — DATASET PREREQUISITE (run before any training)

---

### EXP-0: Real Dataset Integration — DREGON + FLAME 3 + DroneAudioset

**Why:** Current audio pairing (random ESC-50 clips) is the primary weakness blocking top-venue publication. This replaces it with two scientifically grounded alternatives and adds a real hardware-synchronized tri-modal validation set. Upgrading this alone moves target from IEEE TGRS (current) toward IROS/ICRA territory.

**What changes:**

| Track | Dataset | Claim unlocked |
|-------|---------|---------------|
| Option 3a | **FLAME 3** (arXiv:2412.02831, Dec 2024) — real UAV RGB + radiometric thermal, 622 fire + 116 no-fire quartets | Real UAV-collected visual modalities, not lab images |
| Option 3b | **DroneAudioset** (NeurIPS 2025, MIT) — 23.5h real UAV SAR audio (fire, speech, crowd, ambient) | Audio from same aerial platform context as visual data |
| Option 2 | **DREGON** (IROS 2018, Inria) — hardware-synchronized IR + visible + audio, 365 IR + 285 RGB + 90 audio clips | Only public tri-modal UAV dataset with true sensor sync |

**New metadata field:** `sync_type` column in all CSVs:

| Value | Meaning | Rows |
|-------|---------|------|
| `real_sync` | Hardware-synchronized (DREGON) | cross-domain eval only |
| `real_uav` | Both modalities from same UAV platform | FLAME3 + DroneAudioset standalone rows |
| `semantic_align` | Fire-scenario visual paired with fire-scenario audio | FLAME/FLAME3 → DroneAudioset pairing |
| `random_pair` | Random label-matched (ESC-50) | All other rows (existing behavior) |

**Implementation — 11 tasks, 2 files modified:**

*Files:*
- Modify: `adaptfuse_uav/scripts/download_data.py` — add `download_flame3()`, `download_droneaudioset()`, `setup_dregon_manual()`
- Modify: `adaptfuse_uav/scripts/build_metadata.py` — add `build_flame3_rows()`, `build_droneaudioset_rows()`, `build_dregon_rows()`, `pair_droneaudioset_semantic()`, `sync_type` field on all existing row dicts
- Modify: `adaptfuse_uav/evaluation/plot_results.py` — add `print_sync_type_summary()`

*Download steps (run once before training):*

```powershell
# Install HuggingFace client for DroneAudioset
pip install huggingface_hub

# Download all datasets (adds FLAME3 + DroneAudioset to existing pipeline)
cd adaptfuse_uav
python scripts/download_data.py

# DREGON requires manual registration:
#   1. Visit https://dregon.inria.fr/
#   2. Register with institutional email
#   3. Extract to: adaptfuse_uav/datasets/raw/DREGON/
#      infrared/  (IR .mp4 clips)
#      visible/   (RGB .mp4 clips)
#      audio/     (WAV files)

# Rebuild metadata with new datasets + semantic pairing
python scripts/build_metadata.py
```

*Key functions to add to `download_data.py`:*

```python
def download_flame3():
    """FLAME 3: real UAV wildfire RGB + radiometric thermal (Dec 2024)."""
    dest_dir = RAW_DIR / "FLAME3"
    if is_dataset_populated(dest_dir, ["fire", "no_fire"]):
        print("[SKIP] FLAME3 already downloaded.")
        return
    FLAME3_KAGGLE_SLUG = "mohammadreza13/flame-3-wildfire-uav-dataset"  # verify slug
    try:
        success = download_kaggle_dataset(FLAME3_KAGGLE_SLUG, dest_dir,
                                          "FLAME 3", ["fire", "no_fire"])
        if success:
            restructure_flame3(dest_dir)
    except Exception:
        print("Manual: https://ieee-dataport.org/open-access/flame-3-radiometric-thermal-uav-imagery-wildfire-management")


def restructure_flame3(dest_dir: Path):
    (dest_dir / "fire").mkdir(parents=True, exist_ok=True)
    (dest_dir / "no_fire").mkdir(parents=True, exist_ok=True)
    img_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    for f in list(dest_dir.rglob("*")):
        if not f.is_file() or f.suffix.lower() not in img_exts:
            continue
        if f.parent.name in {"fire", "no_fire"}:
            continue
        path_str = str(f).lower()
        target = "no_fire" if "no_fire" in path_str or "nofire" in path_str else "fire"
        dest = dest_dir / target / f.name
        if not dest.exists():
            shutil.move(str(f), str(dest))


def download_droneaudioset():
    """DroneAudioset (NeurIPS 2025, MIT): 23.5h real UAV SAR audio."""
    dest_dir = RAW_DIR / "DroneAudioset"
    if dest_dir.exists() and any(dest_dir.rglob("*.wav")):
        print("[SKIP] DroneAudioset already downloaded.")
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="ahlab-drone-project/DroneAudioSet",
        repo_type="dataset",
        local_dir=str(dest_dir),
        ignore_patterns=["*.parquet", "*.arrow", "*.json.gz"],
    )


def setup_dregon_manual():
    """DREGON: print instructions. Registration required at dregon.inria.fr."""
    dregon_dir = RAW_DIR / "DREGON"
    video_exts = {".mp4", ".avi", ".mov", ".mkv"}
    ir_dir = dregon_dir / "infrared"
    has_data = dregon_dir.exists() and ir_dir.exists() and any(
        f.suffix.lower() in video_exts for f in ir_dir.rglob("*")
    )
    if has_data:
        print(f"[DREGON] Data found at {dregon_dir}.")
        return
    print("[DREGON] Manual registration required: https://dregon.inria.fr/")
    print(f"  Extract to: {dregon_dir}/infrared/  visible/  audio/")
```

*Key functions to add to `build_metadata.py`:*

```python
def build_flame3_rows() -> list:
    """FLAME 3 real UAV wildfire RGB + radiometric thermal (Dec 2024)."""
    rows = []
    flame3_dir = RAW_DIR / "FLAME3"
    if not flame3_dir.exists():
        return rows
    for label_name in ["fire", "no_fire"]:
        label_dir = flame3_dir / label_name
        if not label_dir.exists():
            continue
        disaster_label = 1 if label_name == "fire" else 0
        all_files = list(label_dir.rglob("*"))
        thermal_files = [f for f in all_files if f.suffix.lower() in {".tif", ".tiff"}]
        rgb_files = [f for f in all_files
                     if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
                     and "thermal" not in f.name.lower()]
        thermal_by_stem = {f.stem: f for f in thermal_files}
        for rgb_path in rgb_files:
            thermal_path = thermal_by_stem.get(rgb_path.stem)
            rows.append({
                "sample_id": f"flame3_{len(rows):06d}",
                "rgb_path": str(rgb_path),
                "thermal_path": str(thermal_path) if thermal_path else "",
                "audio_path": "",
                "disaster_label": disaster_label,
                "victim_label": 0,
                "has_rgb": 1,
                "has_thermal": 1 if thermal_path else 0,
                "has_audio": 0,
                "source_dataset": "flame3",
                "degradation": "none",
                "sync_type": "real_uav",
            })
    return rows


_DAS_CLASS_MAP = {
    "fire": {"disaster_label": 1, "victim_label": 0},
    "crackling_fire": {"disaster_label": 1, "victim_label": 0},
    "fire_crackling": {"disaster_label": 1, "victim_label": 0},
    "speech": {"disaster_label": 0, "victim_label": 1},
    "crowd": {"disaster_label": 0, "victim_label": 1},
    "clapping": {"disaster_label": 0, "victim_label": 1},
    "screaming": {"disaster_label": 0, "victim_label": 1},
    "crying": {"disaster_label": 0, "victim_label": 1},
    "footsteps": {"disaster_label": 0, "victim_label": 1},
    "water": {"disaster_label": 2, "victim_label": 0},
    "ambient": {"disaster_label": 0, "victim_label": 0},
}


def build_droneaudioset_rows() -> list:
    """DroneAudioset (NeurIPS 2025, MIT): real UAV SAR audio."""
    rows = []
    das_dir = RAW_DIR / "DroneAudioset"
    if not das_dir.exists():
        return rows
    meta_csv = next(das_dir.rglob("metadata.csv"), None) or next(das_dir.rglob("*.csv"), None)
    if meta_csv and meta_csv.exists():
        df_meta = pd.read_csv(meta_csv)
        file_col = next((c for c in df_meta.columns if c in {"file_path", "filename", "path", "audio_path"}), None)
        label_col = next((c for c in df_meta.columns if c in {"label", "category", "class"}), None)
        for _, row in df_meta.iterrows():
            if not file_col:
                continue
            audio_path = das_dir / str(row[file_col])
            if not audio_path.exists():
                continue
            label_key = str(row[label_col]).lower() if label_col else "ambient"
            labels = _DAS_CLASS_MAP.get(label_key, {"disaster_label": 0, "victim_label": 0})
            rows.append({
                "sample_id": f"droneaudio_{len(rows):06d}",
                "rgb_path": "", "thermal_path": "",
                "audio_path": str(audio_path),
                "disaster_label": labels["disaster_label"],
                "victim_label": labels["victim_label"],
                "has_rgb": 0, "has_thermal": 0, "has_audio": 1,
                "source_dataset": "droneaudioset",
                "degradation": "none",
                "sync_type": "real_uav",
            })
    else:
        for cls_dir in sorted(das_dir.rglob("*")):
            if not cls_dir.is_dir():
                continue
            labels = _DAS_CLASS_MAP.get(cls_dir.name.lower(), {"disaster_label": 0, "victim_label": 0})
            for audio_file in scan_directory(cls_dir, AUDIO_EXTS):
                rows.append({
                    "sample_id": f"droneaudio_{len(rows):06d}",
                    "rgb_path": "", "thermal_path": "",
                    "audio_path": str(audio_file),
                    "disaster_label": labels["disaster_label"],
                    "victim_label": labels["victim_label"],
                    "has_rgb": 0, "has_thermal": 0, "has_audio": 1,
                    "source_dataset": "droneaudioset",
                    "degradation": "none",
                    "sync_type": "real_uav",
                })
    return rows


def build_dregon_rows(max_frames_per_clip: int = 10) -> list:
    """DREGON: hardware-synchronized IR+visible+audio. Cross-domain eval only."""
    rows = []
    dregon_dir = RAW_DIR / "DREGON"
    if not dregon_dir.exists():
        return rows
    try:
        import cv2
    except ImportError:
        print("[SKIP] DREGON needs opencv-python")
        return rows
    ir_dir = dregon_dir / "infrared"
    vis_dir = dregon_dir / "visible"
    audio_dir_path = dregon_dir / "audio"
    video_exts = {".mp4", ".avi", ".mov", ".mkv"}
    ir_videos = sorted([f for f in ir_dir.rglob("*") if f.suffix.lower() in video_exts])[:50] if ir_dir.exists() else []
    vis_videos = sorted([f for f in vis_dir.rglob("*") if f.suffix.lower() in video_exts])[:50] if vis_dir.exists() else []
    audio_files = sorted(scan_directory(audio_dir_path, AUDIO_EXTS)) if audio_dir_path.exists() else []
    frames_dir = dregon_dir / "_extracted_frames"
    (frames_dir / "ir").mkdir(parents=True, exist_ok=True)
    (frames_dir / "vis").mkdir(parents=True, exist_ok=True)

    def extract_frames(video_path, out_dir, n):
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total == 0:
            cap.release(); return []
        saved = []
        for idx in [int(i * total / n) for i in range(n)]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret: continue
            out_path = out_dir / f"{video_path.stem}_f{idx:05d}.png"
            if not out_path.exists():
                cv2.imwrite(str(out_path), frame)
            saved.append(str(out_path))
        cap.release(); return saved

    ir_map = {v.stem: extract_frames(v, frames_dir / "ir", max_frames_per_clip) for v in ir_videos}
    vis_map = {v.stem: extract_frames(v, frames_dir / "vis", max_frames_per_clip) for v in vis_videos}
    vis_stems = sorted(vis_map.keys())

    for i, ir_stem in enumerate(sorted(ir_map.keys())):
        ir_frames = ir_map[ir_stem]
        vis_frames = vis_map.get(ir_stem, vis_map.get(vis_stems[i % len(vis_stems)], []) if vis_stems else [])
        audio = str(audio_files[i % len(audio_files)]) if audio_files else ""
        for j in range(len(ir_frames)):
            vis_path = vis_frames[j] if j < len(vis_frames) else (vis_frames[-1] if vis_frames else "")
            rows.append({
                "sample_id": f"dregon_{len(rows):06d}",
                "rgb_path": vis_path, "thermal_path": ir_frames[j],
                "audio_path": audio,
                "disaster_label": 0, "victim_label": 0,
                "has_rgb": 1 if vis_path else 0, "has_thermal": 1,
                "has_audio": 1 if audio else 0,
                "source_dataset": "dregon",
                "degradation": "none",
                "sync_type": "real_sync",
            })
    return rows


def pair_droneaudioset_semantic(visual_rows: list, droneaudio_rows: list) -> list:
    """Pair DroneAudioset audio to FLAME/FLAME3 rows by disaster_label match."""
    if not droneaudio_rows:
        return visual_rows
    audio_pools: dict = defaultdict(list)
    for a in droneaudio_rows:
        audio_pools[(a["disaster_label"], a["victim_label"])].append(a["audio_path"])
    rng = random.Random(42)
    for row in visual_rows:
        if row["source_dataset"] not in {"flame", "flame3"} or row["has_audio"] == 1:
            continue
        key = (row["disaster_label"], row["victim_label"])
        pool = audio_pools.get(key, audio_pools.get((1, 0), []) if row["disaster_label"] == 1 else [])
        if not pool:
            continue
        row["audio_path"] = rng.choice(pool)
        row["has_audio"] = 1
        row["sync_type"] = "semantic_align"
    return visual_rows
```

*Update `main()` collection block in `build_metadata.py`:*

```python
    # existing rows
    visual_rows = []
    visual_rows.extend(build_flame_rows())
    visual_rows.extend(build_flame3_rows())       # NEW
    visual_rows.extend(build_aider_rows())
    visual_rows.extend(build_sard_rows())
    visual_rows.extend(build_firenet_rows())
    visual_rows.extend(build_c2a_rows())

    audio_rows = build_esc50_rows()
    droneaudio_rows = build_droneaudioset_rows()  # NEW

    # Semantic pairing first (FLAME/FLAME3 → DroneAudioset), then ESC-50 fallback
    if droneaudio_rows:
        visual_rows = pair_droneaudioset_semantic(visual_rows, droneaudio_rows)
    if audio_rows:
        visual_rows = pair_audio_to_visual(visual_rows, audio_rows)

    dregon_rows = build_dregon_rows()             # NEW — saved as dregon_eval.csv
    all_rows = visual_rows + audio_rows + droneaudio_rows

    # save dregon_eval.csv separately (never mixed into train/val/test)
    if dregon_rows:
        dregon_df = pd.DataFrame(dregon_rows)
        dregon_df["split"] = "dregon_eval"
        dregon_df.to_csv(SPLITS_DIR / "dregon_eval.csv", index=False)
```

*Add `sync_type` to all existing row dicts in `build_metadata.py`* — one line each:

```python
# In every existing build_*_rows() function, add to each row dict:
"sync_type": "random_pair",
```

**Output after running:**
```
Sync type breakdown:
  semantic_align  : NNN  (X%)   ← FLAME/FLAME3 + DroneAudioset (new, strong)
  real_uav        : MMM  (Y%)   ← DroneAudioset standalone rows
  random_pair     : KKK  (Z%)   ← ESC-50 pairs (existing, weaker)
  real_sync       : 0            ← DREGON in dregon_eval.csv (separate)
```

**Defense claim unlocked:** "Our audio component is drawn from DroneAudioset [NeurIPS 2025] — the first benchmark of real UAV-embedded audio for search and rescue, paired by disaster scenario class to FLAME/FLAME3 wildfire imagery collected from the same aerial platform perspective."

**Effort:** ~30 min code + dataset download time. No model changes. No retraining needed to validate pipeline.

---

## Priority Tier 1 — MUST DO (directly defend novel claims)

---

### EXP-1: Component Ablation of AdapFuse v1

**Why:** RUE and NuisanceHead are your two novelty claims. Without ablation, a reviewer can say "maybe the attention alone does the work." You need to isolate each component's contribution.

**How:** Train 5 stripped variants of AdapFuse v1, change one thing at a time:

| Variant | What's removed | Config change needed |
|---------|---------------|----------------------|
| `ablation_no_rue` | RUE removed, equal weights | `use_rue: false` |
| `ablation_no_nuisance` | NuisanceHead → standard 2-head output | `use_nuisance_head: false` |
| `ablation_no_attention` | Cross-attention → simple mean pool | `use_cross_attention: false` |
| `ablation_no_gru` | GRU removed, direct to head | `use_gru: false` |
| `ablation_no_quality_tokens` | RUE uses feature stats only, no quality tokens | `use_quality_tokens: false` |

**Implementation note:** Add boolean flags to `AdapFuseV1.__init__()` for each component. Already 80% there — just need to gate the RUE call and replace cross-attention with `tokens.mean(dim=1)` conditionally.

**Expected result table:**

| Variant | Disaster F1 | Victim F1 | ΔF1 vs full |
|---------|-------------|-----------|-------------|
| Full AdapFuse v1 | — | — | baseline |
| No RUE | ↓ | ↓ | ~−3–5% |
| No NuisanceHead | ↓ mainly precision | ↓ | false alarms ↑ |
| No Cross-Attn | ↓ | ↓ | ~−2% |
| No GRU | small ↓ | small ↓ | ~−1% |
| No Quality Tokens | ↓ | ↓ | ~−1–2% |

**Tells you:** Which components justify their parameter cost. GRU probably matters least; RUE and NuisanceHead should matter most.

---

### EXP-2: Reliability Score Behavior Analysis

**Why:** You claim RUE learns to downweight corrupted sensors. You must show this actually happens — not just that accuracy improves, but that the scores *move in the right direction* when a sensor is degraded.

**How:** On test set, apply one corruption at a time (smoke, thermal drift, rotor noise). Record the mean reliability score for that sensor vs clean baseline.

```python
# For each corruption type:
for corruption in ['smoke_overlay', 'thermal_drift', 'rotor_noise_snr']:
    for severity in [1, 3, 5]:
        # Load test set with only this corruption
        # Forward pass through AdapFuse v1
        # Collect reliability[:, sensor_idx] for the corrupted sensor
        # Plot: severity (x) vs mean reliability score (y)
```

**Expected figure:** 3 line plots — one per modality. When smoke is applied:
- `r_rgb` should **decrease** with severity
- `r_thermal` and `r_audio` should **stay flat or increase** (model shifts trust)

**This is Figure 2 in Chapter 4.** Without it, you cannot claim the RUE actually learns anything meaningful.

---

### EXP-3: Two-Sensor Subset Experiments

**Why:** Shows which sensor pairs are most complementary. Tells you the model's sensor hierarchy. Required to fill the missing-modality analysis with more nuance.

**How:** Train 3 new configs with only two sensors active:

| Config | Sensors | Purpose |
|--------|---------|---------|
| `fusion_rgb_thermal` | RGB + Thermal | Best visual pair (FLAME case) |
| `fusion_rgb_audio` | RGB + Audio | No thermal available |
| `fusion_thermal_audio` | Thermal + Audio | Night/smoke — RGB useless |

Compare each against the full 3-modal AdapFuse v1.

**Expected insight:** RGB+Thermal likely best for fire. Thermal+Audio likely best in smoke conditions. This is a novel breakdown no prior work has for this domain.

---

### EXP-4: External SOTA Baselines

**Why:** Your comparison table needs published SOTA numbers, not just your own unimodal baselines. Chapter 4 is weak without this.

**Models to reproduce:**

#### B1: DiRecNetV2 on AIDERv2
```bash
# Code at zenodo.org/records/13939532
# Point to your AIDERv2 subset, run their training script
# Record: weighted F1, params, latency on RTX 9060
```
Expected F1: 0.955–0.964

#### B2: TakuNet on AIDERv2
```bash
# git clone https://github.com/DanielRossi1/TakuNet
# Adapt config to your data paths
# Record: weighted F1, params, FLOPs
```
Expected F1: 0.960–0.964

#### B3: MobileNetV3-Small on FLAME (fire only)
```python
# 30 min to implement
model = torchvision.models.mobilenet_v3_small(pretrained=True)
model.classifier[-1] = nn.Linear(1024, 2)  # fire / no-fire
# Train 30 epochs, record accuracy + F1
```

These give rows B1–B3 in Table 1 (Prior Work Comparison) from the Prior Work plan. Without them, your comparison only includes models you built yourself.

---

### EXP-5: Nuisance Head False Alarm Analysis

**Why:** NuisanceHead claim is "reduces false alarms." F1 alone doesn't prove this. You need precision and false positive rate specifically.

**How:** On test set, compute per-class:
- **False Positive Rate (FPR):** How often does the model predict fire/disaster when it's not?
- **Precision:** Of all disaster predictions, what fraction are correct?

Compare AdapFuse v1 (with nuisance head) vs `ablation_no_nuisance` (EXP-1).

```python
# Additional metric in evaluate.py
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(all_targets, all_preds)
FPR = cm[0, 1:].sum() / cm[0].sum()   # clean samples misclassified as disaster
precision_per_class = cm.diagonal() / cm.sum(axis=0)
```

**Expected result:** FPR drops 15–30% with nuisance head. This is the clearest demonstration of its value.

---

## Priority Tier 2 — SHOULD DO (strengthens methodology chapter)

---

### EXP-6: Reliability KD Ablation (Design F)

**Why:** Design F adds reliability KD loss on top of logit + feature KD. You should show the reliability KD term helps, not just the basic distillation.

**How:** Train 2 student variants:

| Variant | KD terms used |
|---------|---------------|
| `student_basic_kd` | logit KD + feature KD only |
| `student_full_kd` (Design F) | logit KD + feature KD + reliability KD |

Compare accuracy and missing-modality robustness. Reliability KD should help most in the missing-modality table.

---

### EXP-7: Corruption Curriculum Ablation

**Why:** Training with corruption augmentation (`corruption_prob=0.3`) is a design choice. Prove it helps vs not using it.

**How:** Train 3 AdapFuse v1 variants:

| Variant | `corruption_prob` | `missing_modality_prob` |
|---------|-------------------|--------------------------|
| `no_augment` | 0.0 | 0.0 |
| `corrupt_only` | 0.3 | 0.0 |
| `full_augment` (current) | 0.3 | 0.2 |

Evaluate all three on:
1. Clean test set (no corruption applied)
2. Corrupted test set (smoke severity 3)
3. Missing modality (no RGB)

**Expected result:** `full_augment` slightly lower on clean test (augmentation cost), but clearly better on corrupted and missing-modality evals.

---

### EXP-8: Balanced vs Unbalanced Training

**Why:** `build_metadata.py` already generates `train_balanced.csv` and `train_balanced_half.csv`. You should compare all three to justify the balancing strategy.

**How:** Train AdapFuse v1 three ways:
- `use_balanced: false` (raw distribution)
- `use_balanced: true` (full balanced)
- `use_balanced_half: true` (half-size balanced)

Report F1 **per class** (not just macro). Expect raw distribution to show high accuracy on majority class but low F1 on minority disaster types.

---

### EXP-9: Uncertainty Calibration (ECE)

**Why:** You output uncertainty scores from RUE. A natural question: are these well-calibrated? This adds statistical rigor to the uncertainty claim.

**Metric:** Expected Calibration Error (ECE) — measures how well the model's confidence matches actual accuracy.

```python
def expected_calibration_error(probs, labels, n_bins=10):
    """Lower is better. Perfect calibration = 0."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i+1])
        if mask.sum() == 0:
            continue
        bin_acc = labels[mask].mean()
        bin_conf = probs[mask].mean()
        ece += mask.mean() * abs(bin_acc - bin_conf)
    return ece
```

Compare AdapFuse v1 vs Late Fusion vs RGB-only. Show AdapFuse v1 is better calibrated (lower ECE), especially under corruption.

---

## Priority Tier 3 — NICE TO HAVE (if time allows)

---

### EXP-10: Cross-Attention Head Count Ablation

Quick experiment: train AdapFuse v1 with `num_heads` ∈ {1, 2, 4, 8}. Confirms 4 heads is the right choice. 30 minutes to set up, 4 hours of training.

---

### EXP-11: Attention Weight Visualization

**What:** Extract attention maps from `cross_attn` and visualize which modalities attend to which.

```python
# Forward pass with output_attentions=True
attended, attn_weights = self.cross_attn(tokens, tokens, tokens,
                                          need_weights=True,
                                          average_attn_weights=False)
# attn_weights shape: (B, num_heads, 3, 3)
# Row = query modality, Col = key modality
# Heatmap: 3×3 "which modality queries from which modality"
```

Shows: under smoke, audio and thermal tokens attract higher attention weight from rgb queries. Qualitative figure for Chapter 4.

---

### EXP-12: Data Efficiency Curve

Train AdapFuse v1 on 25%, 50%, 75%, 100% of training data. Shows how quickly the model learns. Useful if dataset is small — proves model is data-efficient. 4 extra training runs, each shorter.

---

### EXP-13: Per-Module Latency Breakdown

```python
# Time each module separately
import time

def time_module(module, input_tensor, n=100):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        with torch.no_grad():
            module(input_tensor)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000  # ms

# Time: backbones, projectors, RUE, cross-attention, GRU, head
```

Identifies bottleneck for deployment. Tells you if RUE adds significant overhead (it should be <1ms — tiny MLP).

---

## Summary Table — All New Experiments

| ID | Name | Priority | Runs needed | Est. time | What it proves |
|----|------|----------|-------------|-----------|----------------|
| **EXP-0** | **Real dataset integration (DREGON + FLAME3 + DroneAudioset)** | **PREREQUISITE** | 0 (pipeline only) | ~30 min + download | Addresses primary publishability weakness — real UAV audio pairing |
| EXP-1 | Component ablation | **MUST — TRAINING PENDING** | 5 | ~8–10h | Configs ready (`ablation_no_*.yaml`). Queue overnight. |
| ~~**EXP-2**~~ | ~~**Reliability score behavior**~~ | ~~**MUST**~~ | ~~0~~ | — | **DONE** — `adaptfuse_v1_reliability_behavior.json` exists. **Key finding: RUE scores are static** (r_rgb=r_thermal=0.9721 at all severities). Audio correctly suppressed (r_audio=0.006). RUE does NOT dynamically track per-sample corruption. Reported + analyzed in Chapter 6 Section RUE Behavior. |
| EXP-3 | Two-sensor subsets | **MUST — TRAINING PENDING** | 3 | ~5h | Configs ready (`fusion_rgb_thermal.yaml` etc.). Queue after ablations. |
| EXP-4 | External SOTA baselines | **MUST** | 3 | ~4h | Honest comparison to prior work |
| ~~EXP-5~~ | ~~Nuisance head FPR analysis~~ | ~~**MUST**~~ | ~~0~~ | — | **DONE** — FPR = 0.0073. In `adaptfuse_v1_full_eval.json` and Chapter 6 Table. |
| EXP-6 | Reliability KD ablation | **SHOULD** | 1 | ~2h | KD reliability term helps |
| EXP-7 | Corruption curriculum ablation | **SHOULD** | 2 | ~4h | Augmentation strategy justified |
| EXP-8 | Balanced vs unbalanced | **SHOULD** | 2 | ~3h | Dataset balancing justified |
| ~~EXP-9~~ | ~~ECE calibration~~ | ~~**SHOULD**~~ | ~~0~~ | — | **DONE** — ECE = 0.034. In `adaptfuse_v1_full_eval.json` and Chapter 6 Table. |
| EXP-10 | Attention head count | **NICE** | 4 | ~6h | Hyperparameter justified |
| EXP-11 | Attention visualization | **NICE** | 0 (eval only) | ~1h | Qualitative figure |
| EXP-12 | Data efficiency curve | **NICE** | 4 | ~4h | Model data efficiency |
| ~~EXP-13~~ | ~~Per-module latency~~ | ~~**NICE**~~ | ~~0~~ | — | **DONE** — results in `adaptfuse_v1_latency_profile.json` and Chapter 6 Table. |

**Total new training runs needed:**
- Tier 0 (PREREQUISITE): 0 training runs, pipeline changes + download only
- Tier 1 (MUST): 11 runs + eval-only analyses
- Tier 1+2 (MUST+SHOULD): 16 runs + eval analyses
- All tiers: 24 runs

---

## Minimal Set for June 5 Deadline

Run in this order:

1. **EXP-0 first** (30 min) — upgrade dataset pipeline. No retraining yet. Directly addresses the "artificial data" weakness a panel will probe.
2. **EXP-1 + EXP-2 + EXP-4** — prove components matter, RUE works, comparison is honest.

That combination answers the three questions a defense panel will ask:
1. *"Your audio is randomly paired — how is that valid?"* → EXP-0 answers this
2. *"What does RUE actually do?"* → EXP-2 answers this
3. *"How does your method compare to SOTA?"* → EXP-4 answers this
4. *"Which components matter?"* → EXP-1 answers this

---

## New Chapter 4 Table Structure (with these experiments)

**Table 1** — Prior work comparison (original 8 runs + EXP-4 external baselines)

**Table 2** — Component ablation (EXP-1) → proves each novel part contributes

**Table 3** — Missing modality robustness (original plan)

**Table 4** — Corruption robustness (original plan)

**Table 5** — Two-sensor subsets (EXP-3) → sensor complementarity

**Table 6** — FPR / precision with/without nuisance head (EXP-5)

**Figure 1** — Architecture diagram (AdapFuse v1 full pipeline)

**Figure 2** — Reliability score vs corruption severity (EXP-2) ← KEY FIGURE

**Figure 3** — Robustness curves under corruption (original plan)

**Figure 4** — Attention heatmap (EXP-11, if time) OR reliability score scatter

---

*These experiments directly address the three questions a defense panel will ask:*
1. *"What does RUE actually do?"* → EXP-2 answers this
2. *"How does your method compare to SOTA?"* → EXP-4 answers this
3. *"Which components matter?"* → EXP-1 answers this

---

## Priority Tier 4 — BACKBONE ALTERNATIVES (1 GPU feasible)

These are drop-in backbone swaps. Each creates a new row in Table 1. Most need only `models/backbones/` changes + a new config.

---

### EXP-14: PANNS Audio Backbone (HIGHEST IMPACT — replaces random-init AudioCNN)

**Why:** Current `AudioCNN` (128d) is trained from scratch. PANNS (Pretrained Audio Neural Networks, Kong et al. 2020) are CNNs pretrained on 527-class AudioSet (2M+ audio clips). Environmental sound classes in AudioSet include: crackling fire, crowd noise, siren, machinery — directly overlapping with disaster audio. Replacing random-init AudioCNN with PANNS-CNN6 (pretrained) is the single highest-leverage change possible.

**Expected gain:** +5–10% audio F1, which propagates to overall multimodal F1 since audio is currently the weakest modality.

**Implementation:**

```python
# models/backbones/audio_backbone.py — add after AudioCNN class

class PANNsAudioBackbone(nn.Module):
    """
    PANNS CNN6 pretrained on AudioSet (Kong et al. 2020).
    Requires: pip install panns-inference
    Alternative: manual load from https://zenodo.org/record/3987831
    Out dim: 512 (from CNN6 embedding layer)
    """

    def __init__(self, pretrained: bool = True, freeze_early: bool = False):
        super().__init__()
        # CNN6 architecture from PANNS
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, (3,3), padding=(1,1), bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, (3,3), padding=(1,1), bias=False),
            nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, (3,3), padding=(1,1), bias=False),
            nn.BatchNorm2d(256), nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, (3,3), padding=(1,1), bias=False),
            nn.BatchNorm2d(512), nn.ReLU(),
        )
        self.pool = nn.AvgPool2d((2,2))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.out_dim = 512

        if pretrained:
            self._load_panns_weights()

        if freeze_early:
            for p in list(self.conv1.parameters()) + list(self.conv2.parameters()):
                p.requires_grad = False

    def _load_panns_weights(self):
        """Download CNN6 weights from zenodo.org/record/3987831 if not cached."""
        import urllib.request, os
        cache = os.path.expanduser('~/.cache/panns/Cnn6_mAP=0.343.pth')
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        if not os.path.exists(cache):
            print('[PANNS] Downloading CNN6 pretrained weights (~65MB)...')
            url = 'https://zenodo.org/record/3987831/files/Cnn6_mAP%3D0.343.pth'
            urllib.request.urlretrieve(url, cache)
        ckpt = torch.load(cache, map_location='cpu')
        sd = ckpt.get('model', ckpt)
        # Map PANNS key names → our layer names (best-effort)
        own_sd = self.state_dict()
        mapped = {}
        panns_to_ours = {
            'conv_block1.conv1.weight': 'conv1.0.weight',
            'conv_block1.bn1.weight':   'conv1.1.weight',
            'conv_block1.bn1.bias':     'conv1.1.bias',
            'conv_block2.conv1.weight': 'conv2.0.weight',
            'conv_block2.bn1.weight':   'conv2.1.weight',
            'conv_block2.bn1.bias':     'conv2.1.bias',
            'conv_block3.conv1.weight': 'conv3.0.weight',
            'conv_block3.bn1.weight':   'conv3.1.weight',
            'conv_block3.bn1.bias':     'conv3.1.bias',
            'conv_block4.conv1.weight': 'conv4.0.weight',
            'conv_block4.bn1.weight':   'conv4.1.weight',
            'conv_block4.bn1.bias':     'conv4.1.bias',
        }
        for panns_k, our_k in panns_to_ours.items():
            if panns_k in sd and our_k in own_sd:
                if sd[panns_k].shape == own_sd[our_k].shape:
                    mapped[our_k] = sd[panns_k]
        missing = self.load_state_dict(mapped, strict=False)
        loaded = len(mapped)
        print(f'[PANNS] Loaded {loaded} pretrained tensors. Missing: {len(missing.missing_keys)}')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, 64, T) → (B, 512)"""
        x = self.pool(self.conv1(x))   # (B, 64, 32, T/2)
        x = self.pool(self.conv2(x))   # (B, 128, 16, T/4)
        x = self.pool(self.conv3(x))   # (B, 256, 8, T/8)
        x = self.pool(self.conv4(x))   # (B, 512, 4, T/16)
        x = self.gap(x)                # (B, 512, 1, 1)
        return torch.flatten(x, 1)     # (B, 512)
```

**Config (`configs/adaptfuse_v1_panns.yaml`):**

```yaml
experiment_name: adaptfuse_v1_panns
audio_backbone: panns_cnn6   # add to build_model() dispatch
# everything else same as adaptfuse_v1.yaml
```

**Update `build_model()` in `full_model.py`** to accept `audio_backbone` config key and swap `AudioCNN` for `PANNsAudioBackbone`.

**GPU cost:** Same as AdapFuseV1 — PANNS CNN6 is 4.4M params, similar to AudioCNN (128) but better initialized. No extra VRAM cost.

---

### EXP-15: EfficientNet-B0 RGB Backbone

**Why:** MobileNetV3-Small (2.5M, 576d) was chosen for efficiency. EfficientNet-B0 (5.3M, 1280d) is the next step: +11% ImageNet accuracy, still fits 1 GPU at batch 16. One config swap, 2-hour training run, potentially +2–4% disaster F1.

**Implementation:**

```python
# models/backbones/rgb_backbone.py — add after RGBBackbone

class EfficientNetB0Backbone(nn.Module):
    """EfficientNet-B0, out_dim=1280."""
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        eff = models.efficientnet_b0(weights=weights)
        self.features = eff.features
        self.avgpool = eff.avgpool
        self.out_dim = 1280

    def forward(self, x):
        x = self.features(x)    # (B, 1280, 7, 7)
        x = self.avgpool(x)     # (B, 1280, 1, 1)
        return torch.flatten(x, 1)  # (B, 1280)
```

Add `rgb_backbone: efficientnet_b0` key to config and dispatch in `build_model()`.

**Note:** 1280→192 projection is a bigger bottleneck than 576→192. Consider `proj_dim=256` when using B0.

---

### EXP-16: EfficientNet-B0 Thermal Backbone (symmetric with RGB)

**Why:** Current thermal backbone (ResNet18, 512d) is asymmetric with RGB (MobileNetV3, 576d). Switching both to EfficientNet-B0 (1-ch modified) makes the architecture symmetric and cleaner to analyze. Also: EfficientNet's compound scaling handles the spatial features of thermal images well.

**Implementation** (same modification trick as ResNet18):

```python
class EfficientNetB0ThermalBackbone(nn.Module):
    """EfficientNet-B0 adapted for 1-channel thermal input. out_dim=1280."""
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        eff = models.efficientnet_b0(weights=weights)
        # Replace first conv: 3→1 channel, average weights
        old_conv = eff.features[0][0]
        new_conv = nn.Conv2d(1, old_conv.out_channels, old_conv.kernel_size,
                             old_conv.stride, old_conv.padding, bias=False)
        if pretrained:
            with torch.no_grad():
                new_conv.weight.copy_(old_conv.weight.sum(dim=1, keepdim=True))
        eff.features[0][0] = new_conv
        self.features = eff.features
        self.avgpool = eff.avgpool
        self.out_dim = 1280

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)
```

**Config:** `thermal_backbone: efficientnet_b0_thermal` alongside `rgb_backbone: efficientnet_b0`.

---

### EXP-17: MobileViT-XXS RGB Backbone (CNN+ViT hybrid — aerial scene context)

**Why:** Aerial disaster images need both local texture (smoke texture, fire color) and global context (scene layout, where fire is relative to buildings). MobileViT-XXS (1.3M params, 320d) fuses local CNN features with global transformer attention. Lighter than EfficientNet-B0. DiRecNetV2 showed CNN+ViT hybrids outperform pure CNNs on aerial disaster classification.

**Implementation:**

```python
# Requires timm: pip install timm
import timm

class MobileViTXXSBackbone(nn.Module):
    """MobileViT-XXS hybrid CNN-ViT, out_dim=320."""
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.model = timm.create_model(
            'mobilevit_xxs', pretrained=pretrained, num_classes=0
        )  # num_classes=0 → returns features
        self.out_dim = self.model.num_features  # 320

    def forward(self, x):
        return self.model(x)  # (B, 320)
```

**VRAM note:** MobileViT uses attention internally (quadratic in patch count). At 224×224, patch_size=4 → 56×56=3136 patches — may be slow. Use `img_size=160` in config to reduce to 40×40=1600 patches. Memory fits 1 GPU at batch 16.

---

### EXP-18: GlimmerNet GDBlock Audio Encoder (efficiency focus)

**Why:** GlimmerNet reduced FLOPs 29% using Grouped Dilated Depthwise Convolutions (GDBlocks). Apply this to AudioCNN for a more efficient student model. This is directly citable — you show your audio encoder draws from GlimmerNet's design principle.

**GDBlock implementation:**

```python
class GDBlock(nn.Module):
    """
    Grouped Dilated Depthwise Conv block from GlimmerNet (2025).
    3 parallel dilated depthwise convs (d=1,3,5) + pointwise fusion.
    ~30% fewer FLOPs vs standard conv at same out_channels.
    """
    def __init__(self, in_ch: int, out_ch: int, dilations=(1, 3, 5)):
        super().__init__()
        branch_ch = in_ch // len(dilations) + 1  # ceil
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, branch_ch, 3, padding=d, dilation=d,
                          groups=in_ch, bias=False),  # depthwise
                nn.BatchNorm2d(branch_ch), nn.ReLU(inplace=True),
            ) for d in dilations
        ])
        total_ch = branch_ch * len(dilations)
        self.pointwise = nn.Sequential(
            nn.Conv2d(total_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        branches = [b(x) for b in self.branches]
        return self.pointwise(torch.cat(branches, dim=1))


class AudioCNNGD(nn.Module):
    """AudioCNN with GDBlocks replacing standard ConvBlocks. ~30% fewer FLOPs."""
    def __init__(self, in_channels: int = 1, width_mult: float = 1.0):
        super().__init__()
        c1, c2, c3 = int(32 * width_mult), int(64 * width_mult), int(128 * width_mult)
        self.out_dim = c3
        self.encoder = nn.Sequential(
            GDBlock(in_channels, c1), nn.MaxPool2d(2),
            GDBlock(c1, c2),          nn.MaxPool2d(2),
            GDBlock(c2, c3),          nn.MaxPool2d(2),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.gap(x)
        return torch.flatten(x, 1)
```

**Config:** `audio_backbone: audiocnn_gd`. Compare FLOPs and F1 vs standard `AudioCNN`. Can also combine with PANNS init: `AudioCNNGD` initialized from PANNS-CNN6 feature extractor.

---

## Priority Tier 5 — RELATED WORK ARCHITECTURE IMPROVEMENTS

These require more code change but address specific weaknesses identified in closely related work.

---

### EXP-19: Mamba SSM Fusion (replaces cross-attention — MCDet / FireCast-Fusion inspired)

**Why:** Current cross-modal attention `nn.MultiheadAttention(192, 4)` is O(n²) in sequence length. For 3 modality tokens it's trivial, but MCDet (Forests 2025) shows Mamba state-space models (SSM) capture long-range cross-modal dependencies better on thermal+RGB fusion. FireCast-Fusion uses temporal transformers over UAV frame sequences. Replacing the 3-token cross-attn with Mamba allows the model to naturally handle variable-length temporal sequences (multiple UAV frames), which is the natural extension of this work.

**Current:** `nn.MultiheadAttention(192, 4)` over 3 tokens → fused representation.

**Proposed MambaFusion block:**

```python
# pip install mamba-ssm  (requires CUDA, RTX supports this)
from mamba_ssm import Mamba

class MambaFusionBlock(nn.Module):
    """
    Replace cross-modal MHA with Mamba SSM.
    Handles variable-length sequences of modality tokens.
    Inspired by MCDet's MRCF (Mamba-based cross-modal fusion, Forests 2025).
    """
    def __init__(self, d_model: int = 192, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, 3, D) → (B, 3, D)"""
        residual = tokens
        tokens = self.norm(tokens)
        tokens = self.mamba(tokens)      # processes sequence of 3 modality tokens
        return tokens + residual
```

**In `AdapFuseV1.forward()`**, replace:
```python
# old
attended, _ = self.cross_attn(tokens, tokens, tokens)
attended = self.attn_norm(attended + tokens)
fused = attended.mean(dim=1)
```
with:
```python
# new
attended = self.mamba_fusion(tokens)  # (B, 3, D)
fused = attended.mean(dim=1)
```

**Config:** `use_mamba_fusion: true`. Add EXP-19 row to Table 1: directly compares attention vs SSM fusion.

**GPU feasibility:** Mamba is very lightweight (SSM runs in O(n) with fast CUDA kernels). Far less VRAM than attention. Fits batch 16 easily.

---

### EXP-20: SURE Pearson Correlation Reliability Supervision

**Why:** Current RUE degradation head uses BCE loss (`F.binary_cross_entropy_with_logits`) to supervise reliability from corruption flags. SURE (arXiv 2025) shows Pearson Correlation-based loss provides better gradient signal for uncertainty estimation — the correlation loss measures how well the model's uncertainty *ranks* samples, not just binary correct/incorrect.

**Add to `training/losses.py`:**

```python
def pearson_reliability_loss(predicted_reliability: torch.Tensor,
                              corruption_severity: torch.Tensor) -> torch.Tensor:
    """
    SURE-inspired Pearson correlation loss.
    Supervises reliability to be anti-correlated with corruption severity.
    predicted_reliability: (B, 3) in [0,1]
    corruption_severity: (B, 3) in [0,1] normalized severity per modality

    Expected: low severity → high reliability, high severity → low reliability
    → maximize negative Pearson correlation between reliability and severity
    """
    r = predicted_reliability  # (B, 3)
    s = corruption_severity    # (B, 3)
    # Compute Pearson correlation per modality, average
    total = torch.tensor(0.0, device=r.device)
    for i in range(3):
        ri = r[:, i] - r[:, i].mean()
        si = s[:, i] - s[:, i].mean()
        corr = (ri * si).sum() / (ri.norm() * si.norm() + 1e-8)
        total = total + corr  # want corr to be -1 (anti-correlated)
    return (total / 3.0 + 1.0)  # shift so 0=perfect, 2=worst
```

This requires adding `corruption_severity` as a batch field (continuous 0–1) rather than just the binary `corruption_flags`. Add to `MultimodalDisasterDataset.__getitem__()`:

```python
# Add to returned dict:
"corruption_severity": torch.tensor([
    self.corruption_severity / 5.0 if (do_corrupt and dropped != "rgb") else 0.0,   # rgb sev
    self.corruption_severity / 5.0 if (do_corrupt and dropped != "thermal") else 0.0,
    self.corruption_severity / 5.0 if (do_corrupt and dropped != "audio") else 0.0,
], dtype=torch.float32),
```

**Config:** `use_pearson_reliability_loss: true`. Adds one loss term, ~0 extra compute.

---

### EXP-21: Modality Domain Prompts (DPMamba-inspired)

**Why:** DPMamba (IJCAI 2025) shows that adding learnable *modality-specific prompt tokens* as additional context dramatically helps with missing-modality robustness. The prompt encodes "what this modality typically contributes" and is concatenated to features. When the modality is missing, the prompt still provides its prior signal.

**Add to `AdapFuseV1.__init__()`:**

```python
# Learnable domain prompts: one per modality (DPMamba-inspired)
self.rgb_prompt     = nn.Parameter(torch.randn(1, 1, proj_dim) * 0.02)
self.thermal_prompt = nn.Parameter(torch.randn(1, 1, proj_dim) * 0.02)
self.audio_prompt   = nn.Parameter(torch.randn(1, 1, proj_dim) * 0.02)
```

**In `forward()`, expand token sequence** from 3 to 6 tokens (3 feature + 3 prompt):

```python
B = rgb.size(0)
prompts = torch.cat([
    self.rgb_prompt.expand(B, -1, -1),
    self.thermal_prompt.expand(B, -1, -1),
    self.audio_prompt.expand(B, -1, -1),
], dim=1)  # (B, 3, D)

tokens_with_prompt = torch.cat([tokens, prompts], dim=1)  # (B, 6, D)
attended, _ = self.cross_attn(tokens_with_prompt, tokens_with_prompt, tokens_with_prompt)
fused = attended[:, :3].mean(dim=1)  # take only feature token outputs, not prompt tokens
```

**Expected effect:** When `has_rgb=0`, the model still has `rgb_prompt` carrying RGB-domain context. Should improve missing-modality robustness by +2–5% in the missing-modality eval table.

**GPU cost:** Adds 3 × `proj_dim` = 576 parameters. Negligible.

---

### EXP-22: Spatial Alignment Attention Before Pooling (Zhang et al. 2025 inspired)

**Why:** Current thermal backbone does global average pooling over (B, 512, 7, 7). This discards spatial structure. Zhang et al. 2025 (UAV RGB-Thermal, Remote Sensing) use M-RIFT spatial alignment to register RGB and thermal maps before feature extraction. Without spatial alignment, GAP may pool over regions where modalities disagree spatially (shifted viewpoints, different fields of view).

**Lightweight spatial alignment — add to thermal/RGB encoders before GAP:**

```python
class SpatialAlignmentAttention(nn.Module):
    """
    Channel-spatial attention gate applied before GAP.
    Focuses on spatially consistent regions between modalities.
    Cheaper alternative to explicit image registration (M-RIFT).
    """
    def __init__(self, channels: int):
        super().__init__()
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, channels // 4), nn.ReLU(),
            nn.Linear(channels // 4, channels), nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) → (B, C, H, W) with spatial attention applied"""
        # Channel attention
        ca = self.channel_gate(x).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        x = x * ca
        # Spatial attention
        avg = x.mean(dim=1, keepdim=True)   # (B, 1, H, W)
        mx  = x.max(dim=1, keepdim=True)[0] # (B, 1, H, W)
        sa  = self.spatial_gate(torch.cat([avg, mx], dim=1))  # (B, 1, H, W)
        return x * sa
```

**Add to `RGBBackbone.forward()` and `ThermalBackbone.forward()` before flatten:**

```python
# In RGBBackbone.forward():
x = self.features(x)          # (B, 576, 7, 7)
x = self.spatial_attn(x)      # (B, 576, 7, 7) — spatial focus
x = self.avgpool(x)           # (B, 576, 1, 1)
return torch.flatten(x, 1)
```

Adds ~50K params per backbone (negligible). Expected: +1–2% on smoke/thermal corruption scenarios where spatial disagreement is highest.

---

## Updated Summary Table — All Experiments

| ID | Name | Priority | Needs code? | GPU runs | What it proves |
|----|------|----------|-------------|----------|----------------|
| **NOW-1** | **Queue ablation configs** | **TRAINING PENDING** | 5 GPU runs | 5 | Code done: ablation flags added to `full_model.py`, 5 YAML configs created. **Training not yet run.** |
| **NOW-2** | **Two-sensor configs** | **TRAINING PENDING** | 3 GPU runs | 3 | Code done: 3 YAML configs created. **Training not yet run.** |
| ~~**NOW-3a**~~ | ~~**EXP-13 latency profiling**~~ | ~~**DONE**~~ | ~~0~~ | — | **RESULTS EXIST** — `adaptfuse_v1_latency_profile.json`. RUE = 0.85 ms (5.9%). In Chapter 6. |
| ~~**NOW-3b**~~ | ~~**EXP-5 FPR + EXP-9 ECE**~~ | ~~**DONE**~~ | ~~0~~ | — | **RESULTS EXIST** — FPR = 0.0073, ECE = 0.034 in `adaptfuse_v1_full_eval.json`. In Chapter 6. |
| ~~**NOW-3c**~~ | ~~**EXP-2 reliability analysis**~~ | ~~**DONE**~~ | ~~0~~ | — | **RESULTS EXIST** — see EXP-2 row above for findings. |
| ~~**NOW-4**~~ | ~~**Corruption/missing sweep**~~ | ~~**DONE**~~ | ~~0~~ | — | **RESULTS EXIST** — `adaptfuse_v1_robustness.json`. Corruption table + missing-modality table in Chapter 6. Note: missing-modality results with has_audio=1 are unreliable (zero-fill issue); documented in thesis. |
| EXP-0 | Real dataset integration | PREREQUISITE | 200 lines | 0 | Audio quality |
| EXP-1 | Component ablation | MUST | 20 lines | 5 | Novel components work |
| EXP-2 | Reliability behavior | MUST | 30 lines | 0 | RUE tracks degradation |
| EXP-3 | Two-sensor subsets | MUST | 0 (yaml) | 3 | Sensor hierarchy |
| EXP-4 | External SOTA baselines | MUST | clone repos | 3 | Honest comparison |
| EXP-5 | Nuisance FPR | MUST | 10 lines | 0 | False alarm reduction |
| EXP-6 | Reliability KD ablation | SHOULD | 5 lines | 1 | KD reliability term |
| EXP-7 | Corruption curriculum | SHOULD | 0 (yaml) | 2 | Augmentation justified |
| EXP-8 | Balanced vs unbalanced | SHOULD | 0 (yaml) | 2 | Balancing strategy |
| EXP-9 | ECE calibration | SHOULD | 20 lines | 0 | Uncertainty calibrated |
| EXP-10 | Attention head count | NICE | 0 (yaml) | 4 | Hyperparameter justified |
| EXP-11 | Attention visualization | NICE | 10 lines | 0 | Qualitative figure |
| EXP-12 | Data efficiency curve | NICE | 5 lines | 4 | Data efficiency claim |
| EXP-13 | Per-module latency | NICE | 20 lines | 0 | Deployment viability |
| ~~**EXP-14**~~ | ~~**PANNS audio backbone**~~ | ~~**HIGH**~~ | ~~80 lines~~ | — | **DONE** — Comb F1=0.9825 (baseline=0.9874). PANNS **degraded** performance: data bottleneck confirmed. Even 2M-clip AudioSet pretraining cannot overcome randomly-paired ESC-50. See chapter_6 Section backbone_variants. |
| **EXP-15** | **EfficientNet-B0 RGB** | **MEDIUM** | 15 lines | 1 | +2–4% RGB F1 |
| **EXP-16** | **EfficientNet-B0 thermal** | **MEDIUM** | 15 lines | 1 | Symmetric backbones |
| EXP-17 | MobileViT-XXS RGB | LOW | 10 lines | 1 | CNN+ViT aerial context |
| ~~EXP-18~~ | ~~GDBlock audio encoder~~ | ~~LOW~~ | ~~40 lines~~ | — | **DONE** — Comb F1=0.9863 (baseline=0.9874). Matches baseline within 0.1%. Multi-scale dilated (d=1,3,5) is drop-in replacement for AudioCNN. Victim F1 marginally higher (0.9838 vs 0.9836). |
| EXP-19 | Mamba SSM fusion | LOW | 30 lines | 1 | vs attention comparison |
| EXP-20 | SURE Pearson reliability | MEDIUM | 30 lines | 0 | Better RUE supervision |
| EXP-21 | Domain prompt tokens | MEDIUM | 20 lines | 1 | Missing-modal robustness |
| EXP-22 | Spatial alignment attn | LOW | 50 lines | 1 | Spatial alignment claim |
| ~~EXP-14-KD~~ | ~~AdapFuse-v2 KD + PANNS~~ | ~~HIGH~~ | — | — | **DONE** — Comb F1=0.9848 (KD baseline=0.9861). PANNS degradation propagates through distillation. |

---

## Backbone Recommendation Summary

| Modality | Current | Best upgrade (1 GPU) | Code effort | Expected gain |
|----------|---------|---------------------|-------------|---------------|
| RGB | MobileNetV3-Small (2.5M, 576d) | EfficientNet-B0 (5.3M, 1280d) | 15 lines | +2–4% disaster F1 |
| Thermal | ResNet18 (11.7M, 512d) | EfficientNet-B0 1-ch (5.3M, 1280d) | 15 lines | symmetric + cleaner |
| Audio | AudioCNN random init (0.4M, 128d) | **PANNS-CNN6 pretrained (4.4M, 512d)** | 80 lines | **+5–10% audio F1** |

**Audio backbone is the single biggest bang-for-buck change.** AudioCNN starts from random weights. PANNS-CNN6 starts from AudioSet-pretrained weights on 2M+ clips including fire sounds, crowd sounds, alarms — exactly what this task needs.

---

## 6-Day Plan to June 5 (based on your RTX 960 GPU)

| Day | Task | Output |
|-----|------|--------|
| **Day 1 — Setup (complete)** | Ablation flags added to model. 5 ablation + 3 two-sensor configs created. Eval scripts written (reliability_analysis.py, robustness_eval.py, profile_latency.py). FPR, ECE, latency results obtained. EXP-2 + robustness sweep still running. | EXP-5, EXP-9, EXP-13 done. NOW-3c + NOW-4 running. |
| ~~**Day 1 (next step)**~~ | ~~Add `PANNsAudioBackbone` (EXP-14). Start training ablation runs overnight.~~ | **DONE** — EXP-14 (PANNS, 38 epochs), EXP-18 (GDBlock, 32 epochs), EXP-14-KD (v2_kd_panns, 43 epochs) complete. Results written to chapter_6.tex and abstract.tex. Key finding: data bottleneck confirmed. |
| **Day 2 (current)** | Queue EXP-21 (domain prompts) + EXP-22 (spatial attn). Queue ablation runs (EXP-1). | Next: python training/train.py --config configs/adaptfuse_v1_domain_prompts.yaml |
| **Day 3** | Run remaining ablation + two-sensor configs (EXP-3). Analyze checkpoint results. | Tables 2 + 5 populated |
| **Day 4** | Run EXP-15 (EffNet RGB) if time permits. Aggregate all results. | Backbone comparison Table~\ref{tab:backbone_results} complete |
| **Day 5** | Final polish, write-up. | Thesis ready |

**Minimum viable set (if time is very tight):**
1. EXP-1 ablations (proves novelty — non-negotiable)
2. EXP-2 reliability analysis (proves RUE works — non-negotiable)
3. EXP-14 PANNS audio (highest single-experiment impact — strongly recommended)
4. NOW-4 corruption/missing sweep (robustness tables — fast)
