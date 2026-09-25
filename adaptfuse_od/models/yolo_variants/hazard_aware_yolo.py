"""Ultralytics YOLO detector with the lightweight Hazard Prior Gate."""

from __future__ import annotations

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel

from .ultralytics_wrapper import UltralyticsDetector
from models.novel_modules.hazard_prior_gate import HazardAwareStem, attach_hazard_prior_gate


class HazardAwareDetectionTrainer(DetectionTrainer):
    """Build the HPG model inside the trainer, after pretrained weights load."""

    gate_kwargs = {}
    validation_batch_size = None

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Cap only validation batches to avoid Windows WDDM watchdog timeouts."""
        if mode == "val" and self.validation_batch_size is not None:
            batch_size = min(batch_size, int(self.validation_batch_size))
        return super().get_dataloader(dataset_path, batch_size, rank, mode)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = DetectionModel(cfg, nc=self.data["nc"], verbose=verbose)
        source_layers = getattr(weights, "model", None)
        source_has_gate = (
            source_layers is not None
            and len(source_layers) > 0
            and isinstance(source_layers[0], HazardAwareStem)
        )
        if source_has_gate:
            # Resume: mirror the checkpoint structure first so both the gate
            # and wrapped pretrained stem load with exact parameter names.
            attach_hazard_prior_gate(model, **self.gate_kwargs)
            model.load(weights)
        else:
            # Fresh fine-tuning: load ordinary pretrained weights before
            # wrapping layer zero, preserving the original stem parameters.
            if weights:
                model.load(weights)
            attach_hazard_prior_gate(model, **self.gate_kwargs)
        return model


class HazardAwareYOLODetector(UltralyticsDetector):
    """YOLO wrapper that injects HPG without changing YOLO's detection loss."""

    trainer_class = HazardAwareDetectionTrainer

    def _load_model(self):
        super()._load_model()
        gate_cfg = self.config.get("model", {}).get("hazard_prior_gate", {})
        self.gate_kwargs = dict(
            hidden_channels=gate_cfg.get("hidden_channels", 16),
            use_fire_prior=gate_cfg.get("use_fire_prior", True),
            use_smoke_prior=gate_cfg.get("use_smoke_prior", True),
            use_detail_prior=gate_cfg.get("use_detail_prior", True),
            max_residual=gate_cfg.get("max_residual", 0.25),
            gate_stride=gate_cfg.get("gate_stride", 4),
        )
        training_cfg = self.config.get("training", {})
        self.validation_batch_size = int(
            training_cfg.get("validation_batch_size", training_cfg.get("batch_size", 16))
        )
        HazardAwareDetectionTrainer.gate_kwargs = self.gate_kwargs
        HazardAwareDetectionTrainer.validation_batch_size = self.validation_batch_size
        print(
            f"[{self.architecture}] HPG will be attached inside the detection trainer "
            f"(validation batch <= {self.validation_batch_size})"
        )

    def train(self, dataset_yaml: str, output_dir: str, **kwargs):
        HazardAwareDetectionTrainer.gate_kwargs = self.gate_kwargs
        HazardAwareDetectionTrainer.validation_batch_size = self.validation_batch_size
        return super().train(dataset_yaml, output_dir, **kwargs)
