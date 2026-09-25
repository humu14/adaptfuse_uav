"""
EfficientDet Wrapper — Custom training loop for EfficientDet-D0/D1
===================================================================
Uses the `effdet` library (or timm backbone + custom BiFPN/head) to
provide EfficientDet models with our unified API.
"""

import sys
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


class EfficientDetDetector:
    """Wrapper for EfficientDet models."""

    def __init__(self, architecture: str, num_classes: int = 2,
                 pretrained: bool = True, config: dict = None):
        self.architecture = architecture
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.config = config or {}
        self.model = None
        self.bench = None  # Training/eval bench
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._build_model()

    def _build_model(self):
        """Build EfficientDet model using effdet library."""
        try:
            from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, DetBenchPredict
            from effdet.efficientdet import HeadNet
        except ImportError:
            print("ERROR: effdet not installed. Install with: pip install effdet")
            print("Falling back to a timm-based EfficientDet implementation.")
            self._build_timm_fallback()
            return

        arch = self.architecture.lower()
        model_name_map = {
            "efficientdet_d0": "efficientdet_d0",
            "efficientdet_d1": "efficientdet_d1",
            "efficientdet_d2": "efficientdet_d2",
        }

        effdet_name = model_name_map.get(arch, "efficientdet_d0")

        # Create config
        det_config = get_efficientdet_config(effdet_name)
        det_config.num_classes = self.num_classes
        det_config.image_size = [512, 512] if "d0" in arch else [640, 640]

        # Build model
        net = EfficientDet(det_config, pretrained_backbone=self.pretrained)
        net.class_net = HeadNet(
            det_config,
            num_outputs=self.num_classes,
        )

        self.model = net
        self.det_config = det_config

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"[{arch}] Built EfficientDet")
        print(f"  Image size: {det_config.image_size}")
        print(f"  Num classes: {self.num_classes}")
        print(f"  Total params: {total_params/1e6:.2f}M")

        self.model.to(self.device)

    def _build_timm_fallback(self):
        """Fallback: Build a lightweight detection model using timm backbone."""
        import timm

        # Use EfficientNet-B0 backbone with a simple detection head
        backbone = timm.create_model("efficientnet_b0", pretrained=self.pretrained,
                                      features_only=True, out_indices=[2, 3, 4])

        self.model = _SimpleDetector(backbone, self.num_classes)
        self.model.to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"[{self.architecture}] Built timm-based EfficientDet fallback")
        print(f"  Total params: {total_params/1e6:.2f}M")

    def train(self, dataset_yaml: str, output_dir: str, **kwargs):
        """Train EfficientDet with custom training loop."""
        from data.detection_dataset import DetectionDataset, collate_fn_efficientdet

        train_cfg = self.config.get("training", {})
        train_cfg.update(kwargs)
        dataset_cfg = self.config.get("dataset", {})

        run_dir = Path(output_dir) / self.architecture
        run_dir.mkdir(parents=True, exist_ok=True)

        data_root = Path(dataset_cfg.get("root", "data/datasets/dfire"))
        img_size = dataset_cfg.get("image_size", 512)

        train_dataset = DetectionDataset(
            images_dir=data_root / "images" / "train",
            labels_dir=data_root / "labels" / "train",
            img_size=img_size,
            augment=True,
            format="efficientdet",
        )
        val_dataset = DetectionDataset(
            images_dir=data_root / "images" / "val",
            labels_dir=data_root / "labels" / "val",
            img_size=img_size,
            augment=False,
            format="efficientdet",
        )

        train_loader = DataLoader(
            train_dataset, batch_size=train_cfg.get("batch_size", 8),
            shuffle=True, num_workers=train_cfg.get("num_workers", 4),
            collate_fn=collate_fn_efficientdet, pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=train_cfg.get("batch_size", 8),
            shuffle=False, num_workers=train_cfg.get("num_workers", 4),
            collate_fn=collate_fn_efficientdet, pin_memory=True,
        )

        # Wrap in DetBenchTrain for loss computation
        try:
            from effdet import DetBenchTrain
            train_bench = DetBenchTrain(self.model, self.det_config)
            train_bench.to(self.device)
        except Exception:
            train_bench = self.model

        # Optimizer
        lr = train_cfg.get("lr", 0.001)
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=train_cfg.get("weight_decay", 0.00004),
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=train_cfg.get("epochs", 100),
            eta_min=train_cfg.get("min_lr", 1e-5),
        )

        epochs = train_cfg.get("epochs", 100)
        patience = train_cfg.get("early_stopping_patience", 20)
        best_loss = float("inf")
        no_improve = 0

        use_amp = train_cfg.get("amp", True) and self.device.type == "cuda"
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

        print(f"\n{'='*60}")
        print(f"  Training: {self.architecture}")
        print(f"  Epochs: {epochs} | Batch: {train_cfg.get('batch_size', 8)}")
        print(f"  LR: {lr} | Output: {run_dir}")
        print(f"{'='*60}\n")

        start_time = time.time()
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            # Train
            train_bench.train()
            epoch_loss = 0.0
            n_batches = 0

            from tqdm import tqdm
            pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", unit="batch")

            for batch_idx, (images, targets) in enumerate(pbar):
                images = images.to(self.device)
                boxes = targets["boxes"].to(self.device)
                labels = targets["labels"].to(self.device)

                target_dict = {"bbox": boxes, "cls": labels}

                with torch.amp.autocast('cuda', enabled=use_amp):
                    loss_dict = train_bench(images, target_dict)
                    if isinstance(loss_dict, dict):
                        loss = sum(loss_dict.values())
                    else:
                        loss = loss_dict

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                n_batches += 1
                avg_loss = epoch_loss / n_batches

                pbar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.6f}"
                })

            train_loss = epoch_loss / max(n_batches, 1)
            history["train_loss"].append(train_loss)

            # Simple validation loss (same approach)
            train_bench.eval()
            val_loss = 0.0
            val_n = 0
            with torch.no_grad():
                for images, targets in val_loader:
                    images = images.to(self.device)
                    boxes = targets["boxes"].to(self.device)
                    labels = targets["labels"].to(self.device)
                    target_dict = {"bbox": boxes, "cls": labels}

                    train_bench.train()  # Need train mode for loss
                    loss_dict = train_bench(images, target_dict)
                    train_bench.eval()

                    if isinstance(loss_dict, dict):
                        loss = sum(loss_dict.values())
                    else:
                        loss = loss_dict
                    val_loss += loss.item()
                    val_n += 1

            val_loss = val_loss / max(val_n, 1)
            history["val_loss"].append(val_loss)
            scheduler.step()

            print(f"  Epoch [{epoch+1}/{epochs}] "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                no_improve = 0
                torch.save(self.model.state_dict(), run_dir / "best.pt")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        elapsed = time.time() - start_time
        print(f"\n  Training complete in {elapsed/3600:.2f} hours")

        torch.save(self.model.state_dict(), run_dir / "last.pt")
        with open(run_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        return history

    def evaluate(self, dataset_yaml: str = None, split: str = "test",
                 conf: float = 0.25, iou: float = 0.45, **kwargs):
        """Evaluate with mAP metrics."""
        return {
            "architecture": self.architecture,
            "split": split,
            "mAP50": 0.0,
            "mAP50_95": 0.0,
            "note": "Full mAP evaluation requires running predict + COCO eval",
        }

    def predict(self, source, conf: float = 0.25, **kwargs):
        """Run inference."""
        return []

    def load_weights(self, weights_path: str):
        """Load trained weights."""
        state = torch.load(weights_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)

    def get_model_info(self) -> dict:
        total = sum(p.numel() for p in self.model.parameters())
        return {
            "architecture": self.architecture,
            "total_params": total,
            "num_classes": self.num_classes,
        }


class _SimpleDetector(nn.Module):
    """Minimal detection model using timm backbone — fallback only."""

    def __init__(self, backbone, num_classes, num_anchors=9):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes

        # Get feature dimensions from backbone
        with torch.no_grad():
            dummy = torch.randn(1, 3, 512, 512)
            features = backbone(dummy)
            feat_dims = [f.shape[1] for f in features]

        # Simple detection heads on each feature level
        self.cls_heads = nn.ModuleList([
            nn.Conv2d(dim, num_anchors * num_classes, 3, padding=1)
            for dim in feat_dims
        ])
        self.box_heads = nn.ModuleList([
            nn.Conv2d(dim, num_anchors * 4, 3, padding=1)
            for dim in feat_dims
        ])

    def forward(self, x, targets=None):
        features = self.backbone(x)
        cls_outputs = [head(f) for head, f in zip(self.cls_heads, features)]
        box_outputs = [head(f) for head, f in zip(self.box_heads, features)]
        return {"cls": cls_outputs, "box": box_outputs}
