# =============================================================================
# AdapFuse-OD: Configuration System
# =============================================================================
# Central config loader that merges YAML configs with CLI overrides.
# All training, model, and dataset parameters are managed through this.
# =============================================================================

import yaml
import os
import copy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Default configuration — serves as the base for all experiments
DEFAULT_CONFIG = {
    # --- Project ---
    "project_name": "adaptfuse_od",
    "experiment_name": "default",
    "output_dir": "outputs",
    "seed": 42,

    # --- Dataset ---
    "dataset": {
        "name": "dfire",
        "root": "data/datasets/dfire",
        "num_classes": 2,
        # Official D-Fire class IDs: 0=smoke, 1=fire.
        "class_names": ["smoke", "fire"],
        "image_size": 640,
        "train_split": "train",
        "val_split": "val",
        "test_split": "test",
    },

    # --- Model ---
    "model": {
        "architecture": "yolo26n",
        "pretrained": True,
        "num_classes": 2,
        # Novel modules (Phase 2) — all disabled by default for benchmark
        "use_spd_conv": False,
        "use_seam": False,
        "use_p2_head": False,
        "use_nad_head": False,
        "use_acr": False,
        # NAD head settings (when enabled)
        "nad_num_nuisance_classes": 6,    # genuine_fire, solar_reflection, etc.
        # ACR settings (when enabled)
        "acr_hidden_dim": 64,
        "acr_context_radius": 3,
        "hazard_prior_gate": {
            "hidden_channels": 16,
            "use_fire_prior": True,
            "use_smoke_prior": True,
            "use_detail_prior": True,
            "max_residual": 0.25,
            "gate_stride": 4,
        },
    },

    # --- Training ---
    "training": {
        "epochs": 40,
        "batch_size": 16,
        "optimizer": "AdamW",
        "lr": 0.001,
        "weight_decay": 0.0001,
        "momentum": 0.937,
        "warmup_epochs": 2,
        "scheduler": "cosine",
        "min_lr": 1e-5,
        "grad_accum_steps": 1,
        "early_stopping_patience": 8,
        "amp": True,                      # Automatic Mixed Precision
        "num_workers": 4,
        "pin_memory": True,
        "cache": "disk",
        "close_mosaic": 5,
        "cos_lr": True,
        "save_period": -1,
        "plots": True,
    },

    # --- Augmentation ---
    "augmentation": {
        "mosaic": 0.5,
        "mixup": 0.1,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "flip_lr": 0.5,
        "flip_ud": 0.0,
        "scale": 0.5,
        "translate": 0.1,
        "degrees": 0.0,
        "perspective": 0.0,
        # Corruption augmentations (Phase 2+)
        "smoke_overlay_prob": 0.0,
        "low_light_prob": 0.0,
        "motion_blur_prob": 0.0,
    },

    # --- Loss ---
    "loss": {
        "box_loss_weight": 7.5,
        "cls_loss_weight": 0.5,
        "dfl_loss_weight": 1.5,
        "nad_loss_weight": 0.3,           # Nuisance auxiliary loss weight
    },

    # --- Evaluation ---
    "evaluation": {
        "iou_thresholds": [0.5],          # For mAP@50; extend for mAP@50:95
        "conf_threshold": 0.25,
        "map_conf_threshold": 0.001,
        "nms_iou_threshold": 0.45,
        "max_detections": 300,
        "run_test_after_training": False,
    },
}


def load_config(config_path: str = None, overrides: dict = None) -> dict:
    """Load configuration from YAML file and apply overrides.
    
    Args:
        config_path: Path to YAML config file. If None, uses defaults.
        overrides: Dictionary of override values (e.g., from CLI args).
    
    Returns:
        Merged configuration dictionary.
    """
    config = copy.deepcopy(DEFAULT_CONFIG)

    # Load YAML config if provided
    if config_path is not None:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, "r") as f:
            yaml_config = yaml.safe_load(f) or {}
        
        config = _deep_merge(config, yaml_config)

    # Apply CLI overrides
    if overrides:
        config = _deep_merge(config, overrides)

    # Resolve project-relative paths consistently, regardless of the shell cwd.
    output_dir = Path(config["output_dir"])
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    config["output_dir"] = str(output_dir.resolve())

    dataset_root = Path(config["dataset"]["root"])
    if not dataset_root.is_absolute():
        dataset_root = PROJECT_ROOT / dataset_root
    config["dataset"]["root"] = str(dataset_root.resolve())

    # Post-process: ensure output directory exists
    output_path = output_dir / config["experiment_name"]
    output_path.mkdir(parents=True, exist_ok=True)
    config["output_path"] = str(output_path)

    return config


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def save_config(config: dict, path: str):
    """Save configuration to YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    # Quick test
    cfg = load_config()
    print("Default config loaded successfully.")
    print(f"  Model: {cfg['model']['architecture']}")
    print(f"  Dataset: {cfg['dataset']['name']}")
    print(f"  Epochs: {cfg['training']['epochs']}")
    print(f"  Image size: {cfg['dataset']['image_size']}")
