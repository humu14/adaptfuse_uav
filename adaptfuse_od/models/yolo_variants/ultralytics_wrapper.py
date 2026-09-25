"""
Ultralytics Wrapper — Unified interface for YOLOv8/v9/v10/v11 and RT-DETR
===========================================================================
Wraps the Ultralytics library to provide a consistent train/eval/predict API
that matches our benchmark framework.

Handles:
    - Model loading with COCO pretrained weights
    - Training with our config system
    - Evaluation with mAP metrics
    - Inference with visualization
"""

import os
import sys
import time
import torch
import shutil
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".ultralytics"))


def _validation_batch_trainer(batch_size: int):
    """Create a YOLO trainer that caps validation without changing train batch."""
    from ultralytics.models.yolo.detect import DetectionTrainer

    class CappedValidationTrainer(DetectionTrainer):
        validation_batch_size = int(batch_size)

        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
            if mode == "val":
                batch_size = min(batch_size, self.validation_batch_size)
            return super().get_dataloader(dataset_path, batch_size, rank, mode)

    return CappedValidationTrainer


class UltralyticsDetector:
    """Wrapper around Ultralytics YOLO/RT-DETR models."""

    def __init__(self, model_name: str, architecture: str, num_classes: int = 2,
                 pretrained: bool = True, config: dict = None):
        """
        Args:
            model_name: Ultralytics model file (e.g., 'yolov8n.pt')
            architecture: Human-readable name (e.g., 'yolov8n')
            num_classes: Number of detection classes
            pretrained: Use COCO pretrained weights
            config: Full experiment config dict
        """
        self.model_name = model_name
        self.architecture = architecture
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.config = config or {}
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the Ultralytics model."""
        from ultralytics import YOLO

        configured_weights = self.config.get("model", {}).get("weights")
        if str(self.model_name).lower().endswith(('.yaml', '.yml')):
            self.model = YOLO(self.model_name)
            if self.pretrained and configured_weights:
                weights_path = Path(configured_weights)
                if not weights_path.is_absolute():
                    weights_path = PROJECT_ROOT / weights_path
                if not weights_path.is_file():
                    raise FileNotFoundError(f"Configured model weights not found: {weights_path}")
                self.model.load(str(weights_path.resolve()))
                print(
                    f"[{self.architecture}] Loaded architecture {self.model_name} "
                    f"with transferred weights: {weights_path.resolve()}"
                )
            else:
                print(f"[{self.architecture}] Loaded model architecture: {self.model_name}")
            return

        if self.pretrained:
            # Load with pretrained COCO weights
            self.model = YOLO(self.model_name)
            print(f"[{self.architecture}] Loaded pretrained model: {self.model_name}")
        else:
            # Load architecture only (no weights)
            # For YOLO, .yaml defines architecture, .pt loads weights
            yaml_name = self.model_name.replace(".pt", ".yaml")
            self.model = YOLO(yaml_name)
            print(f"[{self.architecture}] Loaded model architecture: {yaml_name}")

    def train(self, dataset_yaml: str, output_dir: str, **kwargs):
        """
        Train the model on a dataset.
        
        Args:
            dataset_yaml: Path to dataset.yaml (Ultralytics format)
            output_dir: Directory to save training outputs
            **kwargs: Override training hyperparameters
        """
        # Merge config with kwargs
        train_cfg = dict(self.config.get("training", {}))
        train_cfg.update(kwargs)

        dataset_cfg = self.config.get("dataset", {})
        aug_cfg = self.config.get("augmentation", {})

        # Build Ultralytics training args
        train_args = {
            "data": dataset_yaml,
            "epochs": train_cfg.get("epochs", 100),
            "batch": train_cfg.get("batch_size", 16),
            "imgsz": dataset_cfg.get("image_size", 640),
            "optimizer": train_cfg.get("optimizer", "AdamW"),
            "lr0": train_cfg.get("lr", 0.001),
            "lrf": train_cfg.get("min_lr", 1e-5) / max(train_cfg.get("lr", 0.001), 1e-8),
            "weight_decay": train_cfg.get("weight_decay", 0.0001),
            "momentum": train_cfg.get("momentum", 0.937),
            "warmup_epochs": train_cfg.get("warmup_epochs", 5),
            "patience": train_cfg.get("early_stopping_patience", 20),
            "workers": train_cfg.get("num_workers", 4),
            "amp": train_cfg.get("amp", True),
            "project": output_dir,
            "name": self.config.get("experiment_name", self.architecture),
            "exist_ok": True,
            "verbose": True,
            "seed": self.config.get("seed", 42),
            "save": True,
            "save_period": train_cfg.get("save_period", -1),
            "plots": train_cfg.get("plots", True),
            # Augmentation
            "mosaic": aug_cfg.get("mosaic", 0.5),
            "mixup": aug_cfg.get("mixup", 0.1),
            "hsv_h": aug_cfg.get("hsv_h", 0.015),
            "hsv_s": aug_cfg.get("hsv_s", 0.7),
            "hsv_v": aug_cfg.get("hsv_v", 0.4),
            "flipud": aug_cfg.get("flip_ud", 0.0),
            "fliplr": aug_cfg.get("flip_lr", 0.5),
            "scale": aug_cfg.get("scale", 0.5),
            "translate": aug_cfg.get("translate", 0.1),
            "degrees": aug_cfg.get("degrees", 0.0),
            "perspective": aug_cfg.get("perspective", 0.0),
        }

        optional_args = {
            "cache": train_cfg.get("cache"),
            "close_mosaic": train_cfg.get("close_mosaic"),
            "cos_lr": train_cfg.get("cos_lr"),
            "fraction": train_cfg.get("fraction"),
            "device": train_cfg.get("device"),
            "resume": train_cfg.get("resume"),
        }
        train_args.update({k: v for k, v in optional_args.items() if v is not None})

        print(f"\n{'='*60}")
        print(f"  Training: {self.architecture}")
        print(f"  Dataset: {dataset_yaml}")
        print(f"  Epochs: {train_args['epochs']}")
        print(f"  Batch: {train_args['batch']}")
        print(f"  Image size: {train_args['imgsz']}")
        print(f"  Optimizer: {train_args['optimizer']}")
        print(f"  LR: {train_args['lr0']}")
        print(f"  Output: {output_dir}/{self.architecture}")
        print(f"{'='*60}\n")

        start_time = time.time()
        trainer_class = getattr(self, "trainer_class", None)
        validation_batch_size = train_cfg.get("validation_batch_size")
        if trainer_class is None and validation_batch_size is not None:
            trainer_class = _validation_batch_trainer(validation_batch_size)
        if trainer_class is not None:
            train_args["trainer"] = trainer_class
        results = self.model.train(**train_args)
        elapsed = time.time() - start_time

        print(f"\n  Training complete in {elapsed/3600:.2f} hours")
        return results

    def evaluate(self, dataset_yaml: str = None, split: str = "test",
                 conf: float = 0.25, iou: float = 0.45, **kwargs):
        """
        Evaluate the model on a dataset split.
        
        Args:
            dataset_yaml: Path to dataset.yaml
            split: Which split to evaluate ('val' or 'test')
            conf: Confidence threshold
            iou: IoU threshold for NMS
        
        Returns:
            Dictionary with mAP@50, mAP@50:95, per-class AP, etc.
        """
        eval_args = {
            "conf": conf,
            "iou": iou,
            "split": split,
            "verbose": True,
            "plots": True,
        }
        evaluation_cfg = self.config.get("evaluation", {})
        if evaluation_cfg.get("batch_size") is not None:
            eval_args["batch"] = int(evaluation_cfg["batch_size"])
        if dataset_yaml:
            eval_args["data"] = dataset_yaml

        print(f"\n  Evaluating {self.architecture} on {split} split...")
        results = self.model.val(**eval_args)

        # Extract key metrics
        metrics = {
            "architecture": self.architecture,
            "split": split,
            "mAP50": float(results.box.map50) if hasattr(results.box, 'map50') else 0.0,
            "mAP50_95": float(results.box.map) if hasattr(results.box, 'map') else 0.0,
            "precision": float(results.box.mp) if hasattr(results.box, 'mp') else 0.0,
            "recall": float(results.box.mr) if hasattr(results.box, 'mr') else 0.0,
        }

        # Per-class AP
        if hasattr(results.box, 'ap50'):
            ap50_per_class = results.box.ap50
            class_names = self.config.get("dataset", {}).get("class_names", ["smoke", "fire"])
            for i, name in enumerate(class_names):
                if i < len(ap50_per_class):
                    metrics[f"AP50_{name}"] = float(ap50_per_class[i])

        return metrics

    def predict(self, source, conf: float = 0.25, iou: float = 0.45,
                save: bool = True, **kwargs):
        """
        Run inference on images/video.
        
        Args:
            source: Path to image, directory, or video
            conf: Confidence threshold
            iou: IoU threshold for NMS
            save: Whether to save annotated outputs
        """
        results = self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            save=save,
            verbose=True,
            **kwargs,
        )
        return results

    def export(self, format: str = "onnx", **kwargs):
        """Export model to deployment format (ONNX, TorchScript, etc.)."""
        return self.model.export(format=format, **kwargs)

    def load_weights(self, weights_path: str):
        """Load trained weights from a checkpoint."""
        from ultralytics import YOLO
        self.model = YOLO(weights_path)
        print(f"[{self.architecture}] Loaded weights: {weights_path}")

    def get_model_info(self) -> dict:
        """Get model architecture information."""
        info = self.model.info(verbose=False)
        return {
            "architecture": self.architecture,
            "model_name": self.model_name,
            "num_params": sum(p.numel() for p in self.model.model.parameters()),
            "num_classes": self.num_classes,
        }

    @property
    def device(self):
        """Get the device the model is on."""
        return next(self.model.model.parameters()).device if self.model else "cpu"
