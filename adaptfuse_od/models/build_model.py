"""
Model Factory — Unified Interface for All 8 Detector Architectures
===================================================================
Creates, trains, and evaluates any of the 8 benchmark models through
a single consistent API. This abstraction allows fair comparison.

Supported architectures:
    YOLO family (via Ultralytics):
        - yolov8n, yolov8s, yolov8m
        - yolov9t, yolov9s
        - yolov10n, yolov10s
        - yolov11n, yolov11s
    Transformer (via Ultralytics):
        - rtdetr_r18, rtdetr_r50
    TorchVision two-stage:
        - fasterrcnn_mobilenetv3
        - fasterrcnn_resnet50
    TorchVision single-stage:
        - ssd_mobilenetv3
        - ssdlite_mobilenetv3
    EfficientDet (via effdet/timm):
        - efficientdet_d0, efficientdet_d1

Usage:
    from models.build_model import build_model
    model = build_model("yolov8n", num_classes=2, pretrained=True)
"""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Architecture registry
# ---------------------------------------------------------------------------
ULTRALYTICS_MODELS = {
    "yolov8n", "yolov8s", "yolov8m", "yolov8l",
    "yolov9t", "yolov9s", "yolov9m",
    "yolov10n", "yolov10s", "yolov10m",
    "yolov11n", "yolov11s", "yolov11m",
    "yolo26n", "yolo26s", "yolo26m", "yolo26s_p2",
    "rtdetr_r18", "rtdetr_r50", "rtdetr_r101",
    "hazard_yolov10n", "hazard_yolo26n",
}

TORCHVISION_MODELS = {
    "fasterrcnn_mobilenetv3", "fasterrcnn_resnet50",
    "ssd_mobilenetv3", "ssdlite_mobilenetv3",
}

EFFICIENTDET_MODELS = {
    "efficientdet_d0", "efficientdet_d1", "efficientdet_d2",
}

ALL_MODELS = ULTRALYTICS_MODELS | TORCHVISION_MODELS | EFFICIENTDET_MODELS


def build_model(architecture: str, num_classes: int = 2, pretrained: bool = True,
                config: dict = None):
    """
    Build a detector model by name.
    
    Args:
        architecture: Model architecture name (e.g., 'yolov8n', 'fasterrcnn_mobilenetv3')
        num_classes: Number of object classes (default: 2 for fire/smoke)
        pretrained: Whether to use pretrained weights
        config: Full config dict for advanced settings (novel modules, etc.)
    
    Returns:
        A model wrapper with unified train/eval/predict interface.
    """
    arch = architecture.lower().strip()

    if arch in ULTRALYTICS_MODELS:
        return _build_ultralytics_model(arch, num_classes, pretrained, config)
    elif arch in TORCHVISION_MODELS:
        return _build_torchvision_model(arch, num_classes, pretrained, config)
    elif arch in EFFICIENTDET_MODELS:
        return _build_efficientdet_model(arch, num_classes, pretrained, config)
    else:
        raise ValueError(
            f"Unknown architecture: '{arch}'\n"
            f"Supported: {sorted(ALL_MODELS)}"
        )


# ---------------------------------------------------------------------------
# Ultralytics YOLO / RT-DETR builder
# ---------------------------------------------------------------------------
def _build_ultralytics_model(arch: str, num_classes: int, pretrained: bool,
                              config: dict = None):
    """Build a YOLO or RT-DETR model via Ultralytics."""
    if arch in {"hazard_yolov10n", "hazard_yolo26n"}:
        from models.yolo_variants.hazard_aware_yolo import HazardAwareYOLODetector

        return HazardAwareYOLODetector(
            model_name="yolo26n.pt" if arch == "hazard_yolo26n" else "yolov10n.pt",
            architecture=arch,
            num_classes=num_classes,
            pretrained=pretrained,
            config=config,
        )

    from models.yolo_variants.ultralytics_wrapper import UltralyticsDetector

    # Map architecture name to Ultralytics model string
    model_map = {
        "yolov8n": "yolov8n.pt",
        "yolov8s": "yolov8s.pt",
        "yolov8m": "yolov8m.pt",
        "yolov8l": "yolov8l.pt",
        "yolov9t": "yolov9t.pt",
        "yolov9s": "yolov9s.pt",
        "yolov9m": "yolov9m.pt",
        "yolov10n": "yolov10n.pt",
        "yolov10s": "yolov10s.pt",
        "yolov10m": "yolov10m.pt",
        "yolov11n": "yolo11n.pt",
        "yolov11s": "yolo11s.pt",
        "yolov11m": "yolo11m.pt",
        "yolo26n": "yolo26n.pt",
        "yolo26s": "yolo26s.pt",
        "yolo26m": "yolo26m.pt",
        "yolo26s_p2": str(Path(__file__).resolve().parent / "yolo_variants" / "yolo26s_p2.yaml"),
        "rtdetr_r18": "rtdetr-l.pt",    # Ultralytics RT-DETR
        "rtdetr_r50": "rtdetr-x.pt",
        "rtdetr_r101": "rtdetr-x.pt",
    }

    model_name = model_map.get(arch)
    if model_name is None:
        raise ValueError(f"No Ultralytics mapping for: {arch}")

    # A domain checkpoint can initialize a short second-stage fine-tune. This
    # avoids restarting from COCO when changing resolution or preprocessing.
    model_cfg = (config or {}).get("model", {}).get("cfg")
    if model_cfg:
        cfg_path = Path(model_cfg)
        if not cfg_path.is_absolute():
            cfg_path = Path(__file__).resolve().parent.parent / cfg_path
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Configured model architecture not found: {cfg_path}")
        model_name = str(cfg_path.resolve())

    configured_weights = (config or {}).get("model", {}).get("weights")
    if configured_weights and not model_cfg and arch != "yolo26s_p2":
        weights_path = Path(configured_weights)
        if not weights_path.is_absolute():
            weights_path = Path(__file__).resolve().parent.parent / weights_path
        if not weights_path.is_file():
            raise FileNotFoundError(f"Configured model weights not found: {weights_path}")
        model_name = str(weights_path.resolve())

    return UltralyticsDetector(
        model_name=model_name,
        architecture=arch,
        num_classes=num_classes,
        pretrained=pretrained,
        config=config,
    )


# ---------------------------------------------------------------------------
# TorchVision Faster R-CNN / SSD builder
# ---------------------------------------------------------------------------
def _build_torchvision_model(arch: str, num_classes: int, pretrained: bool,
                              config: dict = None):
    """Build a Faster R-CNN or SSD model via TorchVision."""
    from models.non_yolo.torchvision_wrapper import TorchVisionDetector

    return TorchVisionDetector(
        architecture=arch,
        num_classes=num_classes,
        pretrained=pretrained,
        config=config,
    )


# ---------------------------------------------------------------------------
# EfficientDet builder
# ---------------------------------------------------------------------------
def _build_efficientdet_model(arch: str, num_classes: int, pretrained: bool,
                               config: dict = None):
    """Build an EfficientDet model via effdet library."""
    from models.non_yolo.efficientdet_wrapper import EfficientDetDetector

    return EfficientDetDetector(
        architecture=arch,
        num_classes=num_classes,
        pretrained=pretrained,
        config=config,
    )


# ---------------------------------------------------------------------------
# Model info helper
# ---------------------------------------------------------------------------
def get_model_info(architecture: str) -> dict:
    """Get metadata about a model architecture."""
    info_db = {
        "yolov8n":  {"family": "YOLO", "params_m": 3.2,  "flops_g": 8.7,  "input": 640},
        "yolov8s":  {"family": "YOLO", "params_m": 11.2, "flops_g": 28.6, "input": 640},
        "yolov9t":  {"family": "YOLO", "params_m": 2.0,  "flops_g": 7.7,  "input": 640},
        "yolov9s":  {"family": "YOLO", "params_m": 7.2,  "flops_g": 26.7, "input": 640},
        "yolov10n": {"family": "YOLO", "params_m": 2.3,  "flops_g": 6.7,  "input": 640},
        "yolov10s": {"family": "YOLO", "params_m": 7.2,  "flops_g": 21.6, "input": 640},
        "yolov11n": {"family": "YOLO", "params_m": 2.6,  "flops_g": 6.5,  "input": 640},
        "yolov11s": {"family": "YOLO", "params_m": 9.4,  "flops_g": 21.5, "input": 640},
        "hazard_yolov10n": {"family": "YOLO+HPG", "params_m": 2.3, "flops_g": 6.7, "input": 512},
        "yolo26n": {"family": "YOLO26", "params_m": 2.57, "flops_g": 6.1, "input": 512},
        "yolo26s": {"family": "YOLO26", "params_m": 10.0, "flops_g": 22.8, "input": 640},
        "yolo26m": {"family": "YOLO26", "params_m": 21.9, "flops_g": 75.4, "input": 640},
        "yolo26s_p2": {"family": "YOLO26-P2", "params_m": 9.8, "flops_g": 28.0, "input": 640},
        "hazard_yolo26n": {"family": "YOLO26+HPG", "params_m": 2.57, "flops_g": 6.1, "input": 512},
        "rtdetr_r18":  {"family": "DETR", "params_m": 20.0, "flops_g": 60.0, "input": 640},
        "rtdetr_r50":  {"family": "DETR", "params_m": 42.0, "flops_g": 136.0, "input": 640},
        "ssd_mobilenetv3":     {"family": "SSD",  "params_m": 3.4,  "flops_g": 1.0, "input": 320},
        "ssdlite_mobilenetv3": {"family": "SSD",  "params_m": 2.1,  "flops_g": 0.6, "input": 320},
        "fasterrcnn_mobilenetv3": {"family": "FRCNN", "params_m": 19.4, "flops_g": 4.5, "input": 640},
        "fasterrcnn_resnet50":    {"family": "FRCNN", "params_m": 41.8, "flops_g": 134.0, "input": 800},
        "efficientdet_d0": {"family": "EfficientDet", "params_m": 3.9, "flops_g": 2.5, "input": 512},
        "efficientdet_d1": {"family": "EfficientDet", "params_m": 6.6, "flops_g": 6.1, "input": 640},
    }
    return info_db.get(architecture.lower(), {"family": "unknown", "params_m": 0, "flops_g": 0, "input": 640})


def list_models():
    """Print all available models with metadata."""
    print(f"\n{'='*80}")
    print(f"  Available Detector Architectures")
    print(f"{'='*80}")
    print(f"  {'Architecture':<28} {'Family':<12} {'Params':<10} {'FLOPs':<10} {'Input':<8}")
    print(f"  {'-'*68}")
    for arch in sorted(ALL_MODELS):
        info = get_model_info(arch)
        print(f"  {arch:<28} {info['family']:<12} {info['params_m']:.1f}M{'':<5} "
              f"{info['flops_g']:.1f}G{'':<5} {info['input']}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    list_models()
