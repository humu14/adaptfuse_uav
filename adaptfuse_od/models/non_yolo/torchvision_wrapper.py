"""
TorchVision Wrapper — Faster R-CNN and SSD with custom training loop
====================================================================
Unlike Ultralytics which has a built-in trainer, TorchVision models need
a manual training loop. This wrapper provides the same train/eval/predict
interface while handling all the boilerplate internally.
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


class TorchVisionDetector:
    """Wrapper for TorchVision detection models (Faster R-CNN, SSD)."""

    def __init__(self, architecture: str, num_classes: int = 2,
                 pretrained: bool = True, config: dict = None):
        self.architecture = architecture
        self.num_classes = num_classes + 1  # +1 for background class
        self.pretrained = pretrained
        self.config = config or {}
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._build_model()

    def _build_model(self):
        """Build the TorchVision detection model."""
        import torchvision
        from torchvision.models.detection import (
            fasterrcnn_mobilenet_v3_large_fpn,
            fasterrcnn_resnet50_fpn_v2,
            ssd300_vgg16,
            ssdlite320_mobilenet_v3_large,
        )
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        from torchvision.models.detection.ssd import SSDClassificationHead

        arch = self.architecture.lower()

        if arch == "fasterrcnn_mobilenetv3":
            weights = "DEFAULT" if self.pretrained else None
            self.model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
            # Replace the classification head for our number of classes
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(
                in_features, self.num_classes
            )
            print(f"[{arch}] Built Faster R-CNN MobileNetV3-Large FPN")
            print(f"  Head input features: {in_features}")

        elif arch == "fasterrcnn_resnet50":
            weights = "DEFAULT" if self.pretrained else None
            self.model = fasterrcnn_resnet50_fpn_v2(weights=weights)
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(
                in_features, self.num_classes
            )
            print(f"[{arch}] Built Faster R-CNN ResNet50 FPN v2")

        elif arch == "ssd_mobilenetv3":
            # SSD with MobileNetV3 via SSDLite320
            weights = "DEFAULT" if self.pretrained else None
            self.model = ssdlite320_mobilenet_v3_large(weights=weights)
            # Replace classification head
            in_channels = [c.in_channels for c in self.model.head.classification_head.module_list]
            num_anchors = self.model.anchor_generator.num_anchors_per_location()
            self.model.head.classification_head = SSDClassificationHead(
                in_channels, num_anchors, self.num_classes
            )
            print(f"[{arch}] Built SSDLite320 MobileNetV3-Large")

        elif arch == "ssdlite_mobilenetv3":
            weights = "DEFAULT" if self.pretrained else None
            self.model = ssdlite320_mobilenet_v3_large(weights=weights)
            in_channels = [c.in_channels for c in self.model.head.classification_head.module_list]
            num_anchors = self.model.anchor_generator.num_anchors_per_location()
            self.model.head.classification_head = SSDClassificationHead(
                in_channels, num_anchors, self.num_classes
            )
            print(f"[{arch}] Built SSDLite320 MobileNetV3-Large")

        else:
            raise ValueError(f"Unknown TorchVision architecture: {arch}")

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  Total params: {total_params/1e6:.2f}M")
        print(f"  Trainable params: {trainable_params/1e6:.2f}M")

        self.model.to(self.device)

    def train(self, dataset_yaml: str, output_dir: str, **kwargs):
        """
        Train using a custom PyTorch training loop.
        
        TorchVision models expect:
            - images: List[Tensor] of shape (C, H, W)
            - targets: List[Dict] with keys 'boxes' (FloatTensor [N, 4]) and 
                       'labels' (Int64Tensor [N])
        """
        from data.detection_dataset import DetectionDataset, collate_fn

        train_cfg = self.config.get("training", {})
        train_cfg.update(kwargs)
        dataset_cfg = self.config.get("dataset", {})

        # Create output directory
        run_dir = Path(output_dir) / self.architecture
        run_dir.mkdir(parents=True, exist_ok=True)

        # Build datasets
        data_root = Path(dataset_cfg.get("root", "data/datasets/dfire"))
        img_size = dataset_cfg.get("image_size", 640)

        train_dataset = DetectionDataset(
            images_dir=data_root / "images" / "train",
            labels_dir=data_root / "labels" / "train",
            img_size=img_size,
            augment=True,
            format="torchvision",
        )
        val_dataset = DetectionDataset(
            images_dir=data_root / "images" / "val",
            labels_dir=data_root / "labels" / "val",
            img_size=img_size,
            augment=False,
            format="torchvision",
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=train_cfg.get("batch_size", 8),
            shuffle=True,
            num_workers=train_cfg.get("num_workers", 4),
            pin_memory=train_cfg.get("pin_memory", True),
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_cfg.get("batch_size", 8),
            shuffle=False,
            num_workers=train_cfg.get("num_workers", 4),
            pin_memory=True,
            collate_fn=collate_fn,
        )

        # Optimizer
        opt_name = train_cfg.get("optimizer", "SGD")
        lr = train_cfg.get("lr", 0.005)
        wd = train_cfg.get("weight_decay", 0.0005)

        params = [p for p in self.model.parameters() if p.requires_grad]
        if opt_name.upper() == "SGD":
            optimizer = optim.SGD(params, lr=lr, momentum=train_cfg.get("momentum", 0.9),
                                  weight_decay=wd)
        else:
            optimizer = optim.AdamW(params, lr=lr, weight_decay=wd)

        # Scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=train_cfg.get("epochs", 100),
            eta_min=train_cfg.get("min_lr", 1e-5),
        )

        # AMP scaler
        use_amp = train_cfg.get("amp", True) and self.device.type == "cuda"
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

        epochs = train_cfg.get("epochs", 100)
        patience = train_cfg.get("early_stopping_patience", 20)
        best_loss = float("inf")
        no_improve = 0

        print(f"\n{'='*60}")
        print(f"  Training: {self.architecture}")
        print(f"  Device: {self.device}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch: {train_cfg.get('batch_size', 8)}")
        print(f"  LR: {lr}")
        print(f"  AMP: {use_amp}")
        print(f"  Output: {run_dir}")
        print(f"{'='*60}\n")

        start_time = time.time()
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            # --- Train ---
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0

            from tqdm import tqdm
            pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", unit="batch")

            for batch_idx, (images, targets) in enumerate(pbar):
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                with torch.amp.autocast('cuda', enabled=use_amp):
                    loss_dict = self.model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += losses.item()
                num_batches += 1
                avg_loss = epoch_loss / num_batches

                pbar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.6f}"
                })

            train_loss = epoch_loss / max(num_batches, 1)
            history["train_loss"].append(train_loss)

            # --- Validate ---
            self.model.eval()
            val_loss = 0.0
            val_batches = 0

            with torch.no_grad():
                for images, targets in val_loader:
                    images = [img.to(self.device) for img in images]
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                    # TorchVision models return losses in train mode
                    self.model.train()
                    loss_dict = self.model(images, targets)
                    self.model.eval()
                    losses = sum(loss for loss in loss_dict.values())

                    val_loss += losses.item()
                    val_batches += 1

            val_loss = val_loss / max(val_batches, 1)
            history["val_loss"].append(val_loss)

            scheduler.step()

            print(f"  Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")

            # Early stopping & checkpointing
            if val_loss < best_loss:
                best_loss = val_loss
                no_improve = 0
                torch.save(self.model.state_dict(), run_dir / "best.pt")
                print(f"    ✓ New best model saved (val_loss={best_loss:.4f})")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"\n  Early stopping at epoch {epoch+1} (patience={patience})")
                    break

            # Periodic checkpoint
            if (epoch + 1) % 10 == 0:
                torch.save(self.model.state_dict(), run_dir / f"epoch_{epoch+1}.pt")

        elapsed = time.time() - start_time
        print(f"\n  Training complete in {elapsed/3600:.2f} hours")

        # Save last model and history
        torch.save(self.model.state_dict(), run_dir / "last.pt")
        with open(run_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        return history

    def evaluate(self, dataset_yaml: str = None, split: str = "test",
                 conf: float = 0.25, iou: float = 0.45, **kwargs):
        """Evaluate using COCO-style mAP."""
        from data.detection_dataset import DetectionDataset, collate_fn
        from evaluation.metrics import compute_map

        dataset_cfg = self.config.get("dataset", {})
        data_root = Path(dataset_cfg.get("root", "data/datasets/dfire"))
        img_size = dataset_cfg.get("image_size", 640)

        test_dataset = DetectionDataset(
            images_dir=data_root / "images" / split,
            labels_dir=data_root / "labels" / split,
            img_size=img_size,
            augment=False,
            format="torchvision",
        )
        test_loader = DataLoader(
            test_dataset, batch_size=4, shuffle=False,
            num_workers=4, collate_fn=collate_fn,
        )

        self.model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for images, targets in test_loader:
                images = [img.to(self.device) for img in images]
                outputs = self.model(images)

                for output, target in zip(outputs, targets):
                    # Filter by confidence
                    keep = output["scores"] >= conf
                    pred = {
                        "boxes": output["boxes"][keep].cpu(),
                        "scores": output["scores"][keep].cpu(),
                        "labels": output["labels"][keep].cpu(),
                    }
                    all_predictions.append(pred)
                    all_targets.append(target)

        # Compute mAP
        metrics = compute_map(all_predictions, all_targets,
                              iou_thresholds=[0.5, 0.75],
                              num_classes=self.num_classes - 1)

        metrics["architecture"] = self.architecture
        metrics["split"] = split
        return metrics

    def predict(self, source, conf: float = 0.25, **kwargs):
        """Run inference on images."""
        import torchvision.transforms.functional as F
        from PIL import Image

        self.model.eval()
        source = Path(source)

        if source.is_file():
            image_paths = [source]
        elif source.is_dir():
            image_paths = sorted(source.glob("*.jpg")) + sorted(source.glob("*.png"))
        else:
            raise ValueError(f"Invalid source: {source}")

        all_results = []
        for img_path in image_paths:
            img = Image.open(img_path).convert("RGB")
            img_tensor = F.to_tensor(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(img_tensor)

            output = outputs[0]
            keep = output["scores"] >= conf
            result = {
                "path": str(img_path),
                "boxes": output["boxes"][keep].cpu().numpy(),
                "scores": output["scores"][keep].cpu().numpy(),
                "labels": output["labels"][keep].cpu().numpy(),
            }
            all_results.append(result)

        return all_results

    def load_weights(self, weights_path: str):
        """Load trained weights."""
        state = torch.load(weights_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        print(f"[{self.architecture}] Loaded weights: {weights_path}")

    def get_model_info(self) -> dict:
        """Get model info."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {
            "architecture": self.architecture,
            "total_params": total,
            "trainable_params": trainable,
            "num_classes": self.num_classes,
            "device": str(self.device),
        }
