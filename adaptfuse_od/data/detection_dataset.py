"""
Detection Dataset — Unified data loading for YOLO-format annotations
=====================================================================
Loads images and YOLO-format labels, and outputs them in the format
required by either TorchVision or EfficientDet models.

YOLO label format (per line): class_id x_center y_center width height
All values normalized to [0, 1].

Supports:
    - TorchVision format: List[Tensor], List[Dict{boxes, labels}]
    - EfficientDet format: Batch tensor, Dict{bbox, cls}
"""

import os
import cv2
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import torchvision.transforms.functional as F


class DetectionDataset(Dataset):
    """
    General-purpose detection dataset that reads YOLO-format annotations
    and outputs in the requested format.
    """

    def __init__(self, images_dir: str, labels_dir: str, img_size: int = 640,
                 augment: bool = False, format: str = "torchvision",
                 class_names: list = None):
        """
        Args:
            images_dir: Path to images directory
            labels_dir: Path to labels directory (YOLO txt files)
            img_size: Target image size (square)
            augment: Whether to apply training augmentations
            format: Output format — 'torchvision' or 'efficientdet'
            class_names: List of class names (for reference)
        """
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.img_size = img_size
        self.augment = augment
        self.format = format
        self.class_names = class_names or ["smoke", "fire"]

        # Find all image files
        self.image_files = []
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            self.image_files.extend(sorted(self.images_dir.glob(f"*{ext}")))

        if len(self.image_files) == 0:
            print(f"WARNING: No images found in {self.images_dir}")

        print(f"  Dataset: {len(self.image_files)} images | "
              f"augment={augment} | size={img_size} | format={format}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Failed to load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        orig_h, orig_w = img.shape[:2]

        # Load labels
        label_path = self.labels_dir / f"{img_path.stem}.txt"
        boxes = []
        labels = []

        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        w = float(parts[3])
                        h = float(parts[4])

                        # Convert YOLO format (normalized xcywh) to absolute xyxy
                        x1 = (x_center - w / 2) * orig_w
                        y1 = (y_center - h / 2) * orig_h
                        x2 = (x_center + w / 2) * orig_w
                        y2 = (y_center + h / 2) * orig_h

                        # Clamp to image bounds
                        x1 = max(0, min(x1, orig_w - 1))
                        y1 = max(0, min(y1, orig_h - 1))
                        x2 = max(x1 + 1, min(x2, orig_w))
                        y2 = max(y1 + 1, min(y2, orig_h))

                        boxes.append([x1, y1, x2, y2])
                        labels.append(cls_id + 1)  # +1 for background class (TorchVision)

        # Resize image
        img = cv2.resize(img, (self.img_size, self.img_size))

        # Scale boxes to resized image
        scale_x = self.img_size / orig_w
        scale_y = self.img_size / orig_h

        scaled_boxes = []
        for box in boxes:
            scaled_boxes.append([
                box[0] * scale_x,
                box[1] * scale_y,
                box[2] * scale_x,
                box[3] * scale_y,
            ])

        # Apply augmentations
        if self.augment:
            img, scaled_boxes, labels = self._augment(img, scaled_boxes, labels)

        # Convert to tensors
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        if len(scaled_boxes) > 0:
            boxes_tensor = torch.tensor(scaled_boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)

        if self.format == "torchvision":
            target = {
                "boxes": boxes_tensor,
                "labels": labels_tensor,
                "image_id": torch.tensor([idx]),
                "area": (boxes_tensor[:, 2] - boxes_tensor[:, 0]) *
                        (boxes_tensor[:, 3] - boxes_tensor[:, 1]) if len(boxes_tensor) > 0
                        else torch.zeros(0),
                "iscrowd": torch.zeros(len(labels_tensor), dtype=torch.int64),
            }
            return img_tensor, target

        elif self.format == "efficientdet":
            # EfficientDet expects boxes in [y1, x1, y2, x2] format
            if len(scaled_boxes) > 0:
                effdet_boxes = boxes_tensor[:, [1, 0, 3, 2]]  # xyxy → yxyx
            else:
                effdet_boxes = torch.zeros((0, 4), dtype=torch.float32)
            target = {
                "boxes": effdet_boxes,
                "labels": labels_tensor,
            }
            return img_tensor, target

        else:
            raise ValueError(f"Unknown format: {self.format}")

    def _augment(self, img, boxes, labels):
        """Apply training augmentations."""
        h, w = img.shape[:2]

        # Random horizontal flip
        if np.random.random() < 0.5:
            img = img[:, ::-1, :].copy()
            new_boxes = []
            for box in boxes:
                x1, y1, x2, y2 = box
                new_boxes.append([w - x2, y1, w - x1, y2])
            boxes = new_boxes

        # Random color jitter
        if np.random.random() < 0.5:
            # HSV augmentation
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 0] = hsv[:, :, 0] * (1 + np.random.uniform(-0.1, 0.1))
            hsv[:, :, 1] = hsv[:, :, 1] * (1 + np.random.uniform(-0.3, 0.3))
            hsv[:, :, 2] = hsv[:, :, 2] * (1 + np.random.uniform(-0.3, 0.3))
            hsv = np.clip(hsv, 0, 255).astype(np.uint8)
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        return img, boxes, labels


def collate_fn(batch):
    """Collate function for TorchVision models (variable-size targets)."""
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


def collate_fn_efficientdet(batch):
    """Collate function for EfficientDet (batch tensor + padded targets)."""
    images = torch.stack([item[0] for item in batch])

    # Pad boxes and labels to the same number of objects
    max_objects = max(len(item[1]["boxes"]) for item in batch)
    max_objects = max(max_objects, 1)  # At least 1

    batch_boxes = torch.zeros(len(batch), max_objects, 4)
    batch_labels = torch.zeros(len(batch), max_objects, dtype=torch.int64)

    for i, item in enumerate(batch):
        n = len(item[1]["boxes"])
        if n > 0:
            batch_boxes[i, :n] = item[1]["boxes"]
            batch_labels[i, :n] = item[1]["labels"]

    targets = {"boxes": batch_boxes, "labels": batch_labels}
    return images, targets


if __name__ == "__main__":
    # Quick test
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/datasets/dfire")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data)

    for split in ["train", "val", "test"]:
        img_dir = data_root / "images" / split
        lbl_dir = data_root / "labels" / split

        if not img_dir.is_dir():
            print(f"  ✗ {split} images not found at {img_dir}")
            continue

        ds = DetectionDataset(img_dir, lbl_dir, img_size=640, format="torchvision")
        print(f"  {split}: {len(ds)} samples")

        if len(ds) > 0 and args.verify:
            img, target = ds[0]
            print(f"    Image shape: {img.shape}")
            print(f"    Boxes: {target['boxes'].shape}")
            print(f"    Labels: {target['labels']}")
