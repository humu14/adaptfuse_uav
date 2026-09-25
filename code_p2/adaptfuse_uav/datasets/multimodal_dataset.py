"""
AdapFuse-UAV: Multimodal Dataset Loader.
Loads RGB, thermal, and audio data from the unified metadata CSV.
"""

import os
import hashlib
import cv2
import numpy as np
import pandas as pd
import librosa
import random
from pathlib import Path
from typing import Optional, Dict, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from datasets.corruptions import (
    apply_rgb_corruption,
    apply_thermal_corruption,
    apply_audio_corruption,
)


# ─────────────────────────────────────────────────────────────────────────────
# Image transforms
# ─────────────────────────────────────────────────────────────────────────────

def get_rgb_transforms(split: str = "train", img_size: int = 224) -> A.Compose:
    if split == "train":
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.RandomRotate90(p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])


def get_rgb_augmented_transforms(img_size: int = 224) -> A.Compose:
    """
    Stronger augmentation for oversampled (minority-class) rows.
    Applied when row['augmented'] == True during training.
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1, p=0.7),
        A.GaussianBlur(blur_limit=(3, 9), p=0.3),
        A.GaussNoise(p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=20, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_thermal_transforms(split: str = "train", img_size: int = 224) -> A.Compose:
    if split == "train":
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
        ])


# ─────────────────────────────────────────────────────────────────────────────
# Audio preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def load_audio_logmel(
    path: str,
    sr: int = 16000,
    duration: float = 2.0,
    n_mels: int = 64,
    hop_length: int = 512,
    n_fft: int = 1024,
) -> np.ndarray:
    """Load audio file and compute log-mel spectrogram. Returns (1, n_mels, T)."""
    try:
        y, _ = librosa.load(path, sr=sr, duration=duration, mono=True)
        target_len = int(sr * duration)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft
        )
        log_mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
        return log_mel[np.newaxis, ...]  # (1, n_mels, T)
    except Exception:
        # Return silent log-mel on failure
        T = int(np.ceil(sr * duration / hop_length)) + 1
        return np.full((1, n_mels, T), -80.0, dtype=np.float32)


def normalize_logmel(log_mel: np.ndarray) -> np.ndarray:
    """Normalize log-mel to zero mean, unit std."""
    mean = log_mel.mean()
    std = log_mel.std() + 1e-6
    return (log_mel - mean) / std


def pad_or_crop_logmel(log_mel: np.ndarray, target_T: int = 63) -> np.ndarray:
    """Pad or crop log-mel to fixed time dimension."""
    T = log_mel.shape[-1]
    if T >= target_T:
        return log_mel[..., :target_T]
    else:
        pad_width = [(0, 0)] * (log_mel.ndim - 1) + [(0, target_T - T)]
        return np.pad(log_mel, pad_width, mode="constant", constant_values=log_mel.min())


# ─────────────────────────────────────────────────────────────────────────────
# Thermal preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_thermal(img: np.ndarray) -> np.ndarray:
    """
    Normalize thermal image to z-score.
    Input: (H, W) or (H, W, 3)
    Output: (1, H, W) float32
    """
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    mean, std = img.mean(), img.std() + 1e-6
    img = (img - mean) / std
    return img[np.newaxis, ...]  # (1, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# Null tensors for missing modalities
# ─────────────────────────────────────────────────────────────────────────────

def null_rgb() -> np.ndarray:
    return np.zeros((3, 224, 224), dtype=np.float32)


def null_thermal() -> np.ndarray:
    return np.zeros((1, 224, 224), dtype=np.float32)


def null_audio(n_mels: int = 64, target_T: int = 63) -> np.ndarray:
    return np.zeros((1, n_mels, target_T), dtype=np.float32)


MAX_BOXES = 50  # max GT boxes per image (padded with -1)


def load_yolo_bboxes(label_path: str, max_boxes: int = MAX_BOXES) -> Tuple[np.ndarray, int]:
    """
    Load YOLO-format label file: each line = class cx cy w h [pose].
    Returns:
        boxes: (max_boxes, 5) float32 array of [class, cx, cy, w, h], padded with -1
        n_boxes: actual number of valid boxes
    """
    boxes = np.full((max_boxes, 5), -1.0, dtype=np.float32)
    if not label_path or not os.path.exists(label_path):
        return boxes, 0
    try:
        n = 0
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls, cx, cy, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                if n < max_boxes:
                    boxes[n] = [cls, cx, cy, w, h]
                    n += 1
        return boxes, n
    except Exception:
        return boxes, 0


# ─────────────────────────────────────────────────────────────────────────────
# Dataset class
# ─────────────────────────────────────────────────────────────────────────────

class MultimodalDisasterDataset(Dataset):
    """
    Multimodal UAV disaster detection dataset.
    Loads RGB, thermal, and audio from metadata CSV.
    Handles missing modalities via zero-fill + flags.
    """

    IMG_SIZE = 224
    N_MELS = 64
    AUDIO_T = 63   # fixed time frames in log-mel

    def __init__(
        self,
        csv_path: str,
        split: str = "train",
        modalities: list = ["rgb", "thermal", "audio"],
        corruption_prob: float = 0.0,
        missing_modality_prob: float = 0.0,
        corruption_type: Optional[str] = None,
        corruption_severity: int = 3,
        seed: int = 42,
        audio_cache_dir: Optional[str] = None,
    ):
        self.split = split
        self.modalities = modalities
        self.corruption_prob = corruption_prob
        self.missing_modality_prob = missing_modality_prob
        self.corruption_type = corruption_type
        self.corruption_severity = corruption_severity
        self.audio_cache_dir = Path(audio_cache_dir) if audio_cache_dir else None

        self.rgb_tf = get_rgb_transforms(split, self.IMG_SIZE)
        # Stronger augmentation applied to oversampled minority-class rows
        self.rgb_aug_tf = get_rgb_augmented_transforms(self.IMG_SIZE) if split == "train" else self.rgb_tf
        self.thermal_tf = get_thermal_transforms(split, self.IMG_SIZE)

        df = pd.read_csv(csv_path)
        # Ensure 'augmented' column exists (backward compatible with non-balanced CSVs)
        if "augmented" not in df.columns:
            df["augmented"] = False
        else:
            df["augmented"] = df["augmented"].astype(bool)
        self.data = df.reset_index(drop=True)

        random.seed(seed)
        np.random.seed(seed)

    def __len__(self):
        return len(self.data)

    def _load_rgb(self, path: str, corrupt: bool = False, use_augmented: bool = False) -> Tuple[np.ndarray, int]:
        if not path or not os.path.exists(path):
            return null_rgb(), 0
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            return null_rgb(), 0
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Corruption
        if corrupt and self.corruption_type in ("smoke_overlay", "motion_blur", "low_light", "gaussian_noise"):
            img = apply_rgb_corruption(img, self.corruption_type, self.corruption_severity)

        # Use stronger augmentation for oversampled minority-class rows
        tf = self.rgb_aug_tf if (use_augmented and self.split == "train") else self.rgb_tf
        augmented = tf(image=img)
        return augmented["image"], 1  # ToTensorV2 returns CHW float32 tensor

    def _load_thermal(self, path: str, corrupt: bool = False) -> Tuple[np.ndarray, int]:
        if not path or not os.path.exists(path):
            return null_thermal(), 0
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                return null_thermal(), 0

        # Augment on raw image BEFORE z-score so RandomBrightnessContrast
        # operates on valid uint8/float data rather than z-scored values.
        aug = self.thermal_tf(image=img)
        img_aug = aug["image"]
        if isinstance(img_aug, torch.Tensor):
            img_aug = img_aug.numpy()

        thermal = preprocess_thermal(img_aug)  # (1, H, W) z-scored float32

        if corrupt and self.corruption_type == "thermal_drift":
            thermal = apply_thermal_corruption(thermal, "thermal_drift", self.corruption_severity)

        return thermal, 1

    def _load_audio(self, path: str, corrupt: bool = False) -> Tuple[np.ndarray, int]:
        if not path or not os.path.exists(path):
            return null_audio(self.N_MELS, self.AUDIO_T), 0
        cache_path = None
        if self.audio_cache_dir is not None:
            key = hashlib.sha1(str(Path(path).resolve()).encode("utf-8")).hexdigest()
            cache_path = self.audio_cache_dir / f"{key}.npy"
        if cache_path is not None and cache_path.exists():
            log_mel = np.load(cache_path, allow_pickle=False)
        else:
            log_mel = load_audio_logmel(path, n_mels=self.N_MELS)
            log_mel = normalize_logmel(log_mel)
            log_mel = pad_or_crop_logmel(log_mel, self.AUDIO_T)

        if corrupt and self.corruption_type in ("rotor_noise_snr", "wind_noise"):
            log_mel = apply_audio_corruption(log_mel, self.corruption_type, self.corruption_severity)

        return log_mel.astype(np.float32), 1

    def __getitem__(self, idx: int) -> Dict:
        row = self.data.iloc[idx]

        disaster_label = int(row["disaster_label"])
        victim_label = int(row["victim_label"])
        is_augmented = bool(row.get("augmented", False))

        # Determine corruption for this sample
        do_corrupt = self.corruption_prob > 0 and random.random() < self.corruption_prob

        # Determine missing modality (only during training)
        do_missing = (
            self.split == "train"
            and self.missing_modality_prob > 0
            and random.random() < self.missing_modality_prob
        )
        dropped = random.choice(["rgb", "thermal", "audio"]) if do_missing else None

        # Load modalities
        if "rgb" in self.modalities:
            rgb_arr, has_rgb = self._load_rgb(
                str(row["rgb_path"]) if row["has_rgb"] else "",
                corrupt=do_corrupt,
                use_augmented=is_augmented,
            )
            if dropped == "rgb":
                rgb_arr, has_rgb = null_rgb(), 0
        else:
            rgb_arr, has_rgb = null_rgb(), 0

        if "thermal" in self.modalities:
            thermal_arr, has_thermal = self._load_thermal(
                str(row["thermal_path"]) if row["has_thermal"] else "",
                corrupt=do_corrupt,
            )
            if dropped == "thermal":
                thermal_arr, has_thermal = null_thermal(), 0
        else:
            thermal_arr, has_thermal = null_thermal(), 0

        if "audio" in self.modalities:
            audio_arr, has_audio = self._load_audio(
                str(row["audio_path"]) if row["has_audio"] else "",
                corrupt=do_corrupt,
            )
            if dropped == "audio":
                audio_arr, has_audio = null_audio(self.N_MELS, self.AUDIO_T), 0
        else:
            audio_arr, has_audio = null_audio(self.N_MELS, self.AUDIO_T), 0

        # Load bounding boxes (YOLO format: class cx cy w h)
        bbox_path = str(row.get("bbox_path", "")) if "bbox_path" in row.index else ""
        has_bbox_col = int(row.get("has_bbox", 0)) if "has_bbox" in row.index else 0
        if has_bbox_col and bbox_path and bbox_path != "nan":
            bbox_arr, n_boxes = load_yolo_bboxes(bbox_path)
        else:
            bbox_arr, n_boxes = np.full((MAX_BOXES, 5), -1.0, dtype=np.float32), 0

        return {
            "rgb": rgb_arr if isinstance(rgb_arr, torch.Tensor) else torch.tensor(rgb_arr, dtype=torch.float32),
            "thermal": torch.tensor(thermal_arr, dtype=torch.float32),
            "audio": torch.tensor(audio_arr, dtype=torch.float32),
            "has_rgb": torch.tensor(has_rgb, dtype=torch.float32),
            "has_thermal": torch.tensor(has_thermal, dtype=torch.float32),
            "has_audio": torch.tensor(has_audio, dtype=torch.float32),
            "disaster_label": torch.tensor(disaster_label, dtype=torch.long),
            "victim_label": torch.tensor(victim_label, dtype=torch.long),
            "bboxes": torch.tensor(bbox_arr, dtype=torch.float32),   # (MAX_BOXES, 5)
            "n_boxes": torch.tensor(n_boxes, dtype=torch.long),
            "has_bbox": torch.tensor(has_bbox_col, dtype=torch.float32),
            "augmented": torch.tensor(is_augmented, dtype=torch.bool),
            "sample_id": str(row["sample_id"]),
            "source": str(row.get("source_dataset", "unknown")),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Custom collate (handles variable-length string fields)
# ─────────────────────────────────────────────────────────────────────────────

def bbox_collate_fn(batch):
    """Default collate but keeps sample_id/source as lists of strings."""
    from torch.utils.data.dataloader import default_collate
    str_keys = {"sample_id", "source"}
    tensor_batch = [{k: v for k, v in b.items() if k not in str_keys} for b in batch]
    collated = default_collate(tensor_batch)
    for k in str_keys:
        collated[k] = [b[k] for b in batch]
    return collated


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    data_dir: str,
    config: dict,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders from config.

    If `config['use_balanced_half']` is True, prefers the smaller
    `*_balanced_half.csv` files (if present). Otherwise if
    `config['use_balanced']` is True it will use the full
    `*_balanced.csv` files. Falls back to `train.csv` / `val.csv`.
    """
    metadata_dir = config.get("metadata_dir")
    if metadata_dir:
        meta_dir = Path(metadata_dir)
        if not meta_dir.is_absolute():
            meta_dir = Path(data_dir) / meta_dir
    else:
        meta_dir = Path(data_dir) / "data" / "metadata"
    if not meta_dir.exists():
        raise FileNotFoundError(f"Metadata directory not found: {meta_dir}")
    modalities = config.get("modalities", ["rgb", "thermal", "audio"])
    batch_size = config.get("batch_size", 32)
    num_workers = config.get("num_workers", 4)
    corruption_prob = config.get("corruption_prob", 0.0)
    missing_prob = config.get("missing_modality_prob", 0.0)
    audio_cache_dir = config.get("audio_cache_dir")
    if audio_cache_dir and not Path(audio_cache_dir).is_absolute():
        audio_cache_dir = str(Path(data_dir) / audio_cache_dir)
    use_balanced = config.get("use_balanced", False)
    use_balanced_half = config.get("use_balanced_half", False)

    # Choose CSV priority: half-balanced -> full-balanced -> original
    if use_balanced_half and (meta_dir / "train_balanced_half.csv").exists():
        train_csv = "train_balanced_half.csv"
    elif use_balanced and (meta_dir / "train_balanced.csv").exists():
        train_csv = "train_balanced.csv"
    else:
        train_csv = "train.csv"

    if use_balanced_half and (meta_dir / "val_balanced_half.csv").exists():
        val_csv = "val_balanced_half.csv"
    elif use_balanced and (meta_dir / "val_balanced.csv").exists():
        val_csv = "val_balanced.csv"
    else:
        val_csv = "val.csv"

    train_ds = MultimodalDisasterDataset(
        csv_path=str(meta_dir / train_csv),
        split="train",
        modalities=modalities,
        corruption_prob=corruption_prob,
        missing_modality_prob=missing_prob,
        audio_cache_dir=audio_cache_dir,
    )
    val_ds = MultimodalDisasterDataset(
        csv_path=str(meta_dir / val_csv),
        split="val",
        modalities=modalities,
        corruption_prob=0.0,
        missing_modality_prob=0.0,
        audio_cache_dir=audio_cache_dir,
    )
    test_ds = MultimodalDisasterDataset(
        csv_path=str(meta_dir / "test.csv"),
        split="test",
        modalities=modalities,
        corruption_prob=0.0,
        missing_modality_prob=0.0,
        audio_cache_dir=audio_cache_dir,
    )

    _persist = num_workers > 0
    _prefetch = 2 if num_workers > 0 else None

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=_persist, prefetch_factor=_prefetch,
        collate_fn=bbox_collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=_persist, prefetch_factor=_prefetch,
        collate_fn=bbox_collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=_persist, prefetch_factor=_prefetch,
        collate_fn=bbox_collate_fn,
    )

    return train_loader, val_loader, test_loader
