"""
Build metadata CSV files for AdapFuse-UAV.
Scans all raw dataset directories and creates unified train/val/test CSVs.

Label schema:
  disaster_label: 0=normal, 1=fire/smoke, 2=collapse/flood, 3=other_disaster
  victim_label:   0=no_person, 1=person
"""

import os
import sys
import random
import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import numpy as np

# ─── Project root ───────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
RAW_DIR           = ROOT / "datasets" / "raw"
PSEUDO_THERMAL_DIR = ROOT / "datasets" / "pseudo_thermal"
PROCESSED_DIR     = ROOT / "datasets" / "processed"
SPLITS_DIR        = ROOT / "data" / "metadata"

SPLITS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ─── Disaster label mapping ──────────────────────────────────────
DISASTER_LABELS = {
    "normal": 0,
    "non-fire": 0,
    "background": 0,
    "fire": 1,
    "fire_smoke": 1,
    "smoke": 1,
    "collapse": 2,
    "collapsed_building": 2,
    "flood": 2,
    "collapse_flood": 2,
    "flooded_areas": 2,
    "traffic_accident": 3,
    "traffic_incident": 3,
    "other_disaster": 3,
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
AUDIO_EXTS = {".wav", ".ogg", ".flac", ".mp3"}

# ESC-50 fire-related classes
ESC50_FIRE_CLASSES = {"crackling_fire"}
ESC50_CLASSES_MAP = {
    "crackling_fire": 1,   # disaster=fire
}


def scan_directory(base_dir: Path, valid_exts: set) -> list:
    """Walk directory and return all matching file paths."""
    files = []
    if not base_dir.exists():
        return files
    for f in base_dir.rglob("*"):
        if f.suffix.lower() in valid_exts:
            files.append(f)
    return files


def build_flame_rows() -> list:
    """Parse FLAME dataset → RGB + thermal pairs."""
    rows = []
    flame_dir = RAW_DIR / "FLAME"
    if not flame_dir.exists():
        print("[SKIP] FLAME directory not found.")
        return rows

    print("\n[BUILD] FLAME dataset...")
    for split in ["Training", "Test"]:
        for label_name in ["fire", "non-fire"]:
            rgb_dir = flame_dir / split / label_name
            thermal_dir = flame_dir / split / "Thermal" / label_name
            if not rgb_dir.exists():
                continue
            rgb_files = scan_directory(rgb_dir, IMG_EXTS)
            for rgb_path in tqdm(rgb_files, desc=f"FLAME/{split}/{label_name}"):
                # Try to find matching thermal
                thermal_path = thermal_dir / rgb_path.name if thermal_dir.exists() else None
                has_thermal = 1 if (thermal_path and thermal_path.exists()) else 0

                rows.append({
                    "sample_id": f"flame_{len(rows):06d}",
                    "rgb_path": str(rgb_path),
                    "thermal_path": str(thermal_path) if has_thermal else "",
                    "audio_path": "",
                    "disaster_label": DISASTER_LABELS.get(label_name, 0),
                    "victim_label": 0,
                    "has_rgb": 1,
                    "has_thermal": has_thermal,
                    "has_audio": 0,
                    "source_dataset": "flame",
                    "degradation": "none",
                })
    print(f"  FLAME: {len(rows)} samples")
    return rows


def build_aider_rows() -> list:
    """Parse AIDER dataset → RGB only."""
    rows = []
    aider_dir = RAW_DIR / "AIDER"
    if not aider_dir.exists():
        print("[SKIP] AIDER directory not found.")
        return rows

    print("\n[BUILD] AIDER dataset...")
    start = 0
    for cls_dir in sorted(aider_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        label_name = cls_dir.name.lower()
        disaster_label = DISASTER_LABELS.get(label_name, 3)
        files = scan_directory(cls_dir, IMG_EXTS)
        for img_path in tqdm(files, desc=f"AIDER/{cls_dir.name}"):
            rows.append({
                "sample_id": f"aider_{start + len(rows):06d}",
                "rgb_path": str(img_path),
                "thermal_path": "",
                "audio_path": "",
                "disaster_label": disaster_label,
                "victim_label": 0,
                "has_rgb": 1,
                "has_thermal": 0,
                "has_audio": 0,
                "source_dataset": "aider",
                "degradation": "none",
            })
    print(f"  AIDER: {len(rows)} samples")
    return rows


def build_esc50_rows() -> list:
    """Parse ESC-50 audio dataset."""
    rows = []
    esc50_dir = RAW_DIR / "ESC50" / "ESC-50-master"
    meta_file = esc50_dir / "meta" / "esc50.csv"
    audio_dir = esc50_dir / "audio"

    if not esc50_dir.exists():
        print("[SKIP] ESC-50 directory not found.")
        return rows

    print("\n[BUILD] ESC-50 dataset...")
    if meta_file.exists():
        df = pd.read_csv(meta_file)
        for _, row in tqdm(df.iterrows(), total=len(df), desc="ESC-50"):
            category = row["category"]
            audio_path = audio_dir / row["filename"]
            if not audio_path.exists():
                continue
            disaster_label = ESC50_CLASSES_MAP.get(category, 0)
            rows.append({
                "sample_id": f"esc50_{len(rows):06d}",
                "rgb_path": "",
                "thermal_path": "",
                "audio_path": str(audio_path),
                "disaster_label": disaster_label,
                "victim_label": 0,
                "has_rgb": 0,
                "has_thermal": 0,
                "has_audio": 1,
                "source_dataset": "esc50",
                "degradation": "none",
            })
    else:
        # Walk without metadata
        for audio_file in tqdm(scan_directory(audio_dir, AUDIO_EXTS), desc="ESC-50"):
            rows.append({
                "sample_id": f"esc50_{len(rows):06d}",
                "rgb_path": "",
                "thermal_path": "",
                "audio_path": str(audio_file),
                "disaster_label": 0,
                "victim_label": 0,
                "has_rgb": 0,
                "has_thermal": 0,
                "has_audio": 1,
                "source_dataset": "esc50",
                "degradation": "none",
            })

    print(f"  ESC-50: {len(rows)} samples")
    return rows


def build_sard_rows() -> list:
    """Parse SARD dataset → victim detection.
    Handles both flat SARD/images/ and split SARD/{train,test,valid}/images/ layouts.
    """
    rows = []
    sard_dir = RAW_DIR / "SARD"
    if not sard_dir.exists():
        print("[SKIP] SARD directory not found.")
        return rows

    print("\n[BUILD] SARD dataset...")

    # Collect all image dirs: prefer split layout, fall back to flat
    image_dirs = []
    for split in ["train", "test", "valid", "val"]:
        split_imgs = sard_dir / split / "images"
        if split_imgs.exists():
            image_dirs.append(split_imgs)
    if not image_dirs:
        flat = sard_dir / "images"
        if flat.exists():
            image_dirs.append(flat)

    if not image_dirs:
        print("[SKIP] SARD: no image directories found.")
        return rows

    for img_dir in image_dirs:
        files = scan_directory(img_dir, IMG_EXTS)
        for img_path in tqdm(files, desc=f"SARD/{img_dir.parent.name}"):
            rows.append({
                "sample_id": f"sard_{len(rows):06d}",
                "rgb_path": str(img_path),
                "thermal_path": "",
                "audio_path": "",
                "disaster_label": 0,
                "victim_label": 1,   # SARD is all victim=1
                "has_rgb": 1,
                "has_thermal": 0,
                "has_audio": 0,
                "source_dataset": "sard",
                "degradation": "none",
            })
    print(f"  SARD: {len(rows)} samples")
    return rows


def build_firenet_rows() -> list:
    """Parse FireNet dataset."""
    rows = []
    firenet_dir = RAW_DIR / "FireNet"
    if not firenet_dir.exists():
        print("[SKIP] FireNet directory not found.")
        return rows

    print("\n[BUILD] FireNet dataset...")
    for cls_dir in sorted(firenet_dir.rglob("*")):
        if not cls_dir.is_dir():
            continue
        label_name = cls_dir.name.lower()
        if label_name not in DISASTER_LABELS:
            continue
        files = scan_directory(cls_dir, IMG_EXTS)
        for img_path in tqdm(files, desc=f"FireNet/{cls_dir.name}"):
            rows.append({
                "sample_id": f"firenet_{len(rows):06d}",
                "rgb_path": str(img_path),
                "thermal_path": "",
                "audio_path": "",
                "disaster_label": DISASTER_LABELS[label_name],
                "victim_label": 0,
                "has_rgb": 1,
                "has_thermal": 0,
                "has_audio": 0,
                "source_dataset": "firenet",
                "degradation": "none",
            })
    print(f"  FireNet: {len(rows)} samples")
    return rows


def build_c2a_rows() -> list:
    """
    Parse C2A dataset (UAV-Enhanced Combination to Application, ICPR 2024).
    All images contain humans in disaster scenarios → victim_label = 1.

    Handles two layouts:
      Layout A (flat): C2A/{fire,flood,collapsed,traffic}/*.jpg
      Layout B (nested): C2A/C2A_Dataset/new_dataset3/{train,test,val}/images/*.png
                         Class encoded in filename prefix (e.g. fire_image*, flood_image*)

    disaster_label mapping:
      fire / fire_image             → 1 (fire/smoke)
      flood / flood_image           → 2 (collapse/flood)
      collapsed / collapsed_building_image → 2 (collapse/flood)
      traffic / traffic_incident_image     → 3 (other_disaster)
    """
    rows = []
    c2a_dir = RAW_DIR / "C2A"
    if not c2a_dir.exists():
        print("[SKIP] C2A directory not found.")
        return rows

    print("\n[BUILD] C2A dataset (victims in disaster scenarios)...")

    # ── Layout A: flat class subdirs ─────────────────────────────
    C2A_DIR_MAP = {
        "fire": 1,
        "flood": 2,
        "collapsed": 2,
        "traffic": 3,
    }
    layout_a_found = False
    for cls_name, disaster_label in C2A_DIR_MAP.items():
        cls_dir = c2a_dir / cls_name
        if not cls_dir.exists():
            continue
        layout_a_found = True
        files = scan_directory(cls_dir, IMG_EXTS)
        for img_path in tqdm(files, desc=f"C2A/{cls_name}"):
            rows.append({
                "sample_id": f"c2a_{len(rows):06d}",
                "rgb_path": str(img_path),
                "thermal_path": "",
                "audio_path": "",
                "disaster_label": disaster_label,
                "victim_label": 1,
                "has_rgb": 1,
                "has_thermal": 0,
                "has_audio": 0,
                "source_dataset": "c2a",
                "degradation": "none",
            })

    if layout_a_found:
        print(f"  C2A (layout A): {len(rows)} samples")
        return rows

    # ── Layout B: nested split/images with class in filename ─────
    # C2A/C2A_Dataset/new_dataset3/{train,test,val}/images/
    C2A_PREFIX_MAP = {
        "fire_image": 1,
        "flood_image": 2,
        "collapsed_building_image": 2,
        "traffic_incident_image": 3,
    }

    nested_base = c2a_dir / "C2A_Dataset" / "new_dataset3"
    if not nested_base.exists():
        # Try any one-level nesting
        for sub in c2a_dir.rglob("train"):
            if (sub / "images").exists():
                nested_base = sub.parent
                break

    splits_found = 0
    for split in ["train", "test", "val"]:
        split_imgs = nested_base / split / "images"
        if not split_imgs.exists():
            continue
        splits_found += 1
        files = scan_directory(split_imgs, IMG_EXTS)
        for img_path in tqdm(files, desc=f"C2A/{split}"):
            fname = img_path.name
            disaster_label = 3  # default: other_disaster
            for prefix, dlabel in C2A_PREFIX_MAP.items():
                if fname.startswith(prefix):
                    disaster_label = dlabel
                    break
            rows.append({
                "sample_id": f"c2a_{len(rows):06d}",
                "rgb_path": str(img_path),
                "thermal_path": "",
                "audio_path": "",
                "disaster_label": disaster_label,
                "victim_label": 1,
                "has_rgb": 1,
                "has_thermal": 0,
                "has_audio": 0,
                "source_dataset": "c2a",
                "degradation": "none",
            })

    if splits_found == 0:
        print("  [SKIP] C2A: no recognized layout found.")
    else:
        print(f"  C2A (layout B, {splits_found} splits): {len(rows)} samples")
    return rows


def build_synthetic_rows() -> list:
    """Parse synthetic generated samples."""
    rows = []
    synth_dir = RAW_DIR / "SYNTHETIC"
    if not synth_dir.exists():
        print("[SKIP] SYNTHETIC directory not found.")
        return rows

    print("\n[BUILD] SYNTHETIC dataset...")
    rgb_base = synth_dir / "rgb"
    thermal_base = synth_dir / "thermal"
    audio_base = synth_dir / "audio"

    for cls_dir in sorted(rgb_base.iterdir()):
        if not cls_dir.is_dir():
            continue
        label_name = cls_dir.name.lower()
        disaster_label = DISASTER_LABELS.get(label_name, 0)
        rgb_files = sorted(scan_directory(cls_dir, IMG_EXTS))

        for rgb_path in tqdm(rgb_files, desc=f"SYNTHETIC/{cls_dir.name}"):
            stem = rgb_path.stem
            thermal_path = thermal_base / cls_dir.name / rgb_path.name
            rows.append({
                "sample_id": f"synth_{len(rows):06d}",
                "rgb_path": str(rgb_path),
                "thermal_path": str(thermal_path) if thermal_path.exists() else "",
                "audio_path": "",
                "disaster_label": disaster_label,
                "victim_label": 0,
                "has_rgb": 1,
                "has_thermal": 1 if thermal_path.exists() else 0,
                "has_audio": 0,
                "source_dataset": "synthetic",
                "degradation": "none",
            })

    # Add synthetic audio rows
    if audio_base.exists():
        for audio_cls_dir in sorted(audio_base.iterdir()):
            if not audio_cls_dir.is_dir():
                continue
            audio_label = 1 if "fire" in audio_cls_dir.name else 0
def build_flame3_rows() -> list:
    """FLAME 3 real UAV wildfire RGB + radiometric thermal (Dec 2024)."""
    rows = []
    flame3_dir = RAW_DIR / "FLAME3"
    if not flame3_dir.exists():
        return rows

    print("\n[BUILD] FLAME 3 (Radiometric Thermal + RGB)...")
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
        for rgb_path in tqdm(rgb_files, desc=f"FLAME3/{label_name}"):
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
    print(f"  FLAME 3: {len(rows)} samples")
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

    print("\n[BUILD] DroneAudioset (Real UAV SAR Audio)...")
    for cls_dir in sorted(das_dir.rglob("*")):
        if not cls_dir.is_dir():
            continue
        labels = _DAS_CLASS_MAP.get(cls_dir.name.lower(), {"disaster_label": 0, "victim_label": 0})
        for audio_file in scan_directory(cls_dir, AUDIO_EXTS):
            rows.append({
                "sample_id": f"droneaudio_{len(rows):06d}",
                "rgb_path": "",
                "thermal_path": "",
                "audio_path": str(audio_file),
                "disaster_label": labels["disaster_label"],
                "victim_label": labels["victim_label"],
                "has_rgb": 0,
                "has_thermal": 0,
                "has_audio": 1,
                "source_dataset": "droneaudioset",
                "degradation": "none",
                "sync_type": "real_uav",
            })
    print(f"  DroneAudioset: {len(rows)} samples")
    return rows


def build_dregon_rows() -> list:
    """DREGON (IROS 2018): hardware-synchronized IR + visible + audio clips."""
    rows = []
    dregon_dir = RAW_DIR / "DREGON"
    if not dregon_dir.exists():
        return rows

    print("\n[BUILD] DREGON (Hardware-Synchronized Tri-Modal)...")
    vis_dir = dregon_dir / "visible"
    ir_dir = dregon_dir / "infrared"
    au_dir = dregon_dir / "audio"

    if vis_dir.exists() and ir_dir.exists():
        vis_files = scan_directory(vis_dir, IMG_EXTS | {".mp4", ".avi"})
        for vf in vis_files:
            ir_f = ir_dir / vf.name
            au_f = au_dir / (vf.stem + ".wav")
            has_ir = 1 if ir_f.exists() else 0
            has_au = 1 if au_f.exists() else 0
            rows.append({
                "sample_id": f"dregon_{len(rows):06d}",
                "rgb_path": str(vf),
                "thermal_path": str(ir_f) if has_ir else "",
                "audio_path": str(au_f) if has_au else "",
                "disaster_label": 0,
                "victim_label": 1 if "human" in vf.name.lower() or "person" in vf.name.lower() else 0,
                "has_rgb": 1,
                "has_thermal": has_ir,
                "has_audio": has_au,
                "source_dataset": "dregon",
                "degradation": "none",
                "sync_type": "real_sync",
            })
    print(f"  DREGON: {len(rows)} samples")
    return rows


def pair_audio_to_visual(rows: list, audio_rows: list) -> list:
    """
    Pair ESC-50 audio samples to visual samples that lack audio.
    Matching by disaster_label.
    """
    print("\n[PAIR] Pairing audio to visual samples...")
    
    # Group audio by disaster_label
    audio_by_label = defaultdict(list)
    for a in audio_rows:
        audio_by_label[a["disaster_label"]].append(a["audio_path"])

    rng = random.Random(42)
    paired_count = 0
    for row in tqdm(rows, desc="Pairing audio"):
        if row["has_audio"] == 0 and row["disaster_label"] in audio_by_label:
            choices = audio_by_label[row["disaster_label"]]
            if choices and rng.random() < 0.6:  # 60% chance of pairing
                row["audio_path"] = rng.choice(choices)
                row["has_audio"] = 1
                paired_count += 1
    
    print(f"  Paired {paired_count} audio samples to visual samples.")
    return rows


def balance_split(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Exact class balance: each disaster class and each victim class get equal counts.

    Stage 1: oversample each disaster class to max_disaster_count (exact equality).
    Stage 2: oversample victim=1 proportionally across disaster classes so
             disaster balance is preserved while victim counts become equal.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["augmented"] = False

    # Stage 1: exact disaster balance
    target_d = int(df["disaster_label"].value_counts().max())
    parts = []
    for cls in sorted(df["disaster_label"].unique()):
        cls_df = df[df["disaster_label"] == cls].copy()
        deficit = target_d - len(cls_df)
        if deficit > 0:
            extra = cls_df.sample(
                n=deficit, replace=True,
                random_state=int(rng.integers(0, 2**31 - 1)),
            ).copy()
            extra["augmented"] = True
            cls_df = pd.concat([cls_df, extra], ignore_index=True)
        parts.append(cls_df)
    balanced = pd.concat(parts, ignore_index=True)

    # Stage 2: exact victim balance — add victim=1 proportionally across disaster classes
    v0 = int((balanced["victim_label"] == 0).sum())
    v1 = int((balanced["victim_label"] == 1).sum())
    to_add = v0 - v1
    if to_add > 0:
        disaster_classes = sorted(balanced["disaster_label"].unique())
        pools = {
            cls: balanced[(balanced["disaster_label"] == cls) & (balanced["victim_label"] == 1)]
            for cls in disaster_classes
        }
        eligible = [cls for cls in disaster_classes if len(pools[cls]) > 0]
        if eligible:
            per_class = to_add // len(eligible)
            remainder = to_add - per_class * len(eligible)
            extra_parts = []
            for i, cls in enumerate(eligible):
                n = per_class + (1 if i < remainder else 0)
                if n == 0:
                    continue
                extra = pools[cls].sample(
                    n=n, replace=True,
                    random_state=int(rng.integers(0, 2**31 - 1)),
                ).copy()
                extra["augmented"] = True
                extra_parts.append(extra)
            if extra_parts:
                balanced = pd.concat([balanced] + extra_parts, ignore_index=True)

    return balanced.sample(frac=1, random_state=seed).reset_index(drop=True)


def stratified_split(df: pd.DataFrame, train_frac=0.7, val_frac=0.15, seed=42):
    """Stratified split by disaster_label."""
    from sklearn.model_selection import train_test_split

    train_rows, temp_rows = train_test_split(
        df, train_size=train_frac, stratify=df["disaster_label"], random_state=seed
    )
    val_rows, test_rows = train_test_split(
        temp_rows,
        train_size=val_frac / (1 - train_frac),
        stratify=temp_rows["disaster_label"],
        random_state=seed,
    )
    train_rows = train_rows.copy()
    val_rows = val_rows.copy()
    test_rows = test_rows.copy()
    train_rows["split"] = "train"
    val_rows["split"] = "val"
    test_rows["split"] = "test"
    return train_rows, val_rows, test_rows


def add_pseudo_thermal_paths(rows: list) -> list:
    """
    For every row that has RGB but no thermal, check whether a pseudo-thermal
    image exists under PSEUDO_THERMAL_DIR (generated by generate_pseudo_thermal.py).
    The pseudo-thermal path mirrors the raw path with suffix changed to .png.

    If PSEUDO_THERMAL_DIR doesn't exist yet, skip silently so this function is
    safe to call before the generation script has been run.
    """
    if not PSEUDO_THERMAL_DIR.exists():
        print("\n[INFO] pseudo_thermal dir not found — run generate_pseudo_thermal.py first.")
        return rows

    linked = 0
    for row in rows:
        if row.get("has_thermal", 0) == 1:
            continue  # already has real thermal
        rgb_path = row.get("rgb_path", "")
        if not rgb_path:
            continue
        try:
            rel = Path(rgb_path).relative_to(RAW_DIR)
        except ValueError:
            continue
        pseudo_path = PSEUDO_THERMAL_DIR / rel.with_suffix(".png")
        if pseudo_path.exists():
            row["thermal_path"] = str(pseudo_path)
            row["has_thermal"] = 1
            linked += 1

    print(f"\n[THERMAL] Linked {linked:,} pseudo-thermal images.")
    return rows


def main(args):
    print("=" * 60)
    print("AdapFuse-UAV: Building Metadata CSVs")
    print("=" * 60)

    random.seed(42)
    np.random.seed(42)

    # ── Collect all rows ──────────────────────────────────────────
    visual_rows = []
    visual_rows.extend(build_flame_rows())
    visual_rows.extend(build_aider_rows())
    visual_rows.extend(build_sard_rows())
    visual_rows.extend(build_firenet_rows())
    visual_rows.extend(build_c2a_rows())  # C2A: victims in disaster scenarios
    visual_rows.extend(build_flame3_rows())  # FLAME 3: real radiometric thermal
    visual_rows.extend(build_dregon_rows())  # DREGON: real synchronized tri-modal

    audio_rows = build_esc50_rows()
    audio_rows.extend(build_droneaudioset_rows())  # DroneAudioset: real UAV SAR audio
    synth_audio_rows = []

    all_audio_for_pairing = audio_rows  # ESC-50 + DroneAudioset for pairing

    # ── Pair audio to visual samples ─────────────────────────────
    if all_audio_for_pairing:
        visual_rows = pair_audio_to_visual(visual_rows, all_audio_for_pairing)

    # ── Combine all rows ─────────────────────────────────────────
    all_rows = visual_rows + audio_rows + synth_audio_rows

    if len(all_rows) == 0:
        print("\n[ERROR] No data found! Run scripts/download_data.py first.")
        sys.exit(1)

    # ── Link pseudo-thermal images ────────────────────────────────
    all_rows = add_pseudo_thermal_paths(all_rows)

    df = pd.DataFrame(all_rows)

    # ── Basic stats ───────────────────────────────────────────────
    print("\n[STATS] Dataset summary:")
    print(f"  Total samples: {len(df)}")

    print(f"\n  Disaster label distribution:")
    for label, count in df["disaster_label"].value_counts().sort_index().items():
        names = {0: "normal", 1: "fire/smoke", 2: "collapse/flood", 3: "other"}
        pct = count / len(df) * 100
        print(f"    Class {label} ({names[label]}): {count}  ({pct:.1f}%)")

    print(f"\n  Victim label distribution:")
    for label, count in df["victim_label"].value_counts().sort_index().items():
        names = {0: "no_victim", 1: "victim_present"}
        pct = count / len(df) * 100
        print(f"    Class {label} ({names[label]}): {count}  ({pct:.1f}%)")

    print(f"\n  Source dataset breakdown:")
    for src, count in df["source_dataset"].value_counts().items():
        pct = count / len(df) * 100
        print(f"    {src:12}: {count}  ({pct:.1f}%)")

    print(f"\n  Joint distribution (disaster_label x victim_label):")
    ct = pd.crosstab(df["disaster_label"], df["victim_label"])
    print(ct.to_string(index=True))

    print(f"\n  Has RGB: {df['has_rgb'].sum()}")
    print(f"  Has Thermal: {df['has_thermal'].sum()}")
    print(f"  Has Audio: {df['has_audio'].sum()}")

    # ── Split ─────────────────────────────────────────────────────
    print("\n[SPLIT] Creating train/val/test splits...")
    train_df, val_df, test_df = stratified_split(df, seed=args.seed)

    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # ── Save ──────────────────────────────────────────────────────
    train_df.to_csv(SPLITS_DIR / "train.csv", index=False)
    val_df.to_csv(SPLITS_DIR / "val.csv", index=False)
    test_df.to_csv(SPLITS_DIR / "test.csv", index=False)

    train_balanced = balance_split(train_df, seed=args.seed)
    val_balanced = balance_split(val_df, seed=args.seed)
    train_balanced.to_csv(SPLITS_DIR / "train_balanced.csv", index=False)
    val_balanced.to_csv(SPLITS_DIR / "val_balanced.csv", index=False)

    # Also save full CSV
    df["split"] = "unassigned"
    df.loc[train_df.index, "split"] = "train"
    df.loc[val_df.index, "split"] = "val"
    df.loc[test_df.index, "split"] = "test"
    df.to_csv(SPLITS_DIR / "all_samples.csv", index=False)

    print(f"\n[DONE] Metadata CSVs saved to: {SPLITS_DIR}")
    print("  - train.csv")
    print("  - val.csv")
    print("  - test.csv")
    print("  - all_samples.csv")
    print("  - train_balanced.csv")
    print("  - val_balanced.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build AdapFuse-UAV metadata CSVs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
