"""Automatic provenance, prediction, metric, and failure-gallery artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


DISASTER_NAMES = {0: "clean", 1: "fire_smoke", 2: "collapse_flood", 3: "other_hazard"}
VICTIM_NAMES = {0: "absent", 1: "present"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_metadata_dir(data_dir: str, config: dict) -> Path:
    configured = config.get("metadata_dir")
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else Path(data_dir) / path
    return Path(data_dir) / "data" / "metadata"


def code_snapshot(root: Path) -> dict:
    tracked = []
    digest = hashlib.sha256()
    for directory in ("configs", "datasets", "evaluation", "models", "scripts", "training"):
        base = root / directory
        if not base.exists():
            continue
        for path in sorted(p for p in base.rglob("*") if p.suffix in {".py", ".yaml", ".yml"}):
            relative = path.relative_to(root).as_posix()
            file_hash = sha256_file(path)
            tracked.append({"path": relative, "sha256": file_hash})
            digest.update(relative.encode("utf-8"))
            digest.update(file_hash.encode("ascii"))

    git_head = None
    git_dirty = None
    try:
        git_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        git_dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True,
            stderr=subprocess.DEVNULL,
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        pass
    return {
        "git_head": git_head,
        "git_dirty": git_dirty,
        "code_tree_sha256": digest.hexdigest(),
        "files": tracked,
    }


def environment_manifest(device: torch.device, seed: int) -> dict:
    packages = {}
    for package in (
        "torch", "torchvision", "numpy", "pandas", "scikit-learn", "opencv-python",
        "librosa", "albumentations", "PyYAML",
    ):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    gpu = None
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        gpu = {
            "name": torch.cuda.get_device_name(device),
            "vram_bytes": int(properties.total_memory),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        }
        try:
            gpu["driver_version"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip().splitlines()[0]
        except (OSError, subprocess.CalledProcessError, IndexError):
            gpu["driver_version"] = None
    return {
        "created_unix": time.time(),
        "random_seed": int(seed),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "device": str(device),
        "cuda": {
            "available": bool(torch.cuda.is_available()),
            "torch_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        },
        "gpu": gpu,
        "packages": packages,
    }


def initialize_run_artifacts(
    output_dir: Path,
    root: Path,
    config: dict,
    config_path: Path,
    data_dir: str,
    device: torch.device,
    started_unix: float | None = None,
) -> dict:
    artifact_dir = output_dir / "run_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = resolve_metadata_dir(data_dir, config).resolve()

    (artifact_dir / "command.txt").write_text(
        subprocess.list2cmdline([sys.executable, *sys.argv]) + "\n", encoding="utf-8"
    )
    with (artifact_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True)
    shutil.copy2(config_path, artifact_dir / "source_config.yaml")
    (artifact_dir / "environment.json").write_text(
        json.dumps(environment_manifest(device, int(config.get("seed", 42))), indent=2),
        encoding="utf-8",
    )
    (artifact_dir / "code_snapshot.json").write_text(
        json.dumps(code_snapshot(root), indent=2), encoding="utf-8"
    )

    split_files = {}
    for name in ("train.csv", "val.csv", "test.csv", "split_report.json"):
        path = metadata_dir / name
        if path.exists():
            split_files[name] = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    aggregate = hashlib.sha256()
    for name, details in sorted(split_files.items()):
        aggregate.update(name.encode("utf-8"))
        aggregate.update(details["sha256"].encode("ascii"))
    split_manifest = {
        "metadata_dir": str(metadata_dir),
        "aggregate_sha256": aggregate.hexdigest(),
        "files": split_files,
    }
    (artifact_dir / "dataset_manifest.json").write_text(
        json.dumps(split_manifest, indent=2), encoding="utf-8"
    )
    report_path = metadata_dir / "split_report.json"
    if report_path.exists():
        shutil.copy2(report_path, artifact_dir / "split_report.json")
    return {
        "artifact_dir": artifact_dir,
        "metadata_dir": metadata_dir,
        "started_unix": started_unix if started_unix is not None else time.time(),
        "seed": int(config.get("seed", 42)),
    }


def refresh_resolved_config(context: dict, config: dict) -> None:
    """Rewrite the resolved config after any runtime-safe fallback is applied."""
    with (context["artifact_dir"] / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True)


def _per_class_rows(predictions: pd.DataFrame, target: str, names: dict[int, str]) -> list[dict]:
    y_true = predictions[f"{target}_gt"].astype(int)
    y_pred = predictions[f"{target}_pred"].astype(int)
    labels = sorted(names)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return [
        {
            "task": target,
            "class_id": int(label),
            "class_name": names[label],
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(labels)
    ]


def save_detailed_metrics(predictions: pd.DataFrame, artifact_dir: Path) -> tuple[Path, Path]:
    class_rows = _per_class_rows(predictions, "disaster", DISASTER_NAMES)
    class_rows += _per_class_rows(predictions, "victim", VICTIM_NAMES)
    per_class_path = artifact_dir / "test_metrics_per_class.csv"
    pd.DataFrame(class_rows).to_csv(per_class_path, index=False)

    source_rows = []
    for source, group in predictions.groupby("source", sort=True):
        d_f1 = f1_score(
            group.disaster_gt,
            group.disaster_pred,
            labels=sorted(DISASTER_NAMES),
            average="macro",
            zero_division=0,
        )
        v_f1 = f1_score(
            group.victim_gt,
            group.victim_pred,
            labels=sorted(VICTIM_NAMES),
            average="macro",
            zero_division=0,
        )
        source_rows.append({
            "source": source,
            "n": int(len(group)),
            "disaster_f1_macro": float(d_f1),
            "victim_f1_macro": float(v_f1),
            "combined_f1": float((d_f1 + v_f1) / 2),
            "disaster_accuracy": float(accuracy_score(group.disaster_gt, group.disaster_pred)),
            "victim_accuracy": float(accuracy_score(group.victim_gt, group.victim_pred)),
        })
    per_source_path = artifact_dir / "test_metrics_per_source.csv"
    pd.DataFrame(source_rows).to_csv(per_source_path, index=False)
    return per_class_path, per_source_path


def save_failure_gallery(
    predictions: pd.DataFrame,
    metadata_dir: Path,
    artifact_dir: Path,
    max_examples: int = 12,
) -> tuple[Path, Path]:
    import cv2
    import matplotlib.pyplot as plt

    metadata = pd.read_csv(metadata_dir / "test.csv")
    lookup = metadata.set_index("sample_id", drop=False)
    failures = predictions[
        predictions.disaster_gt.ne(predictions.disaster_pred)
        | predictions.victim_gt.ne(predictions.victim_pred)
    ].copy()
    failures["error_count"] = (
        failures.disaster_gt.ne(failures.disaster_pred).astype(int)
        + failures.victim_gt.ne(failures.victim_pred).astype(int)
    )
    failures["error_confidence"] = failures.apply(
        lambda row: max(
            row.get(f"disaster_prob_{int(row.disaster_pred)}", 0.0),
            row.get(f"victim_prob_{int(row.victim_pred)}", 0.0),
        ),
        axis=1,
    )
    selected = failures.sort_values(
        ["error_count", "error_confidence", "sample_id"],
        ascending=[False, False, True],
        kind="mergesort",
    ).head(max_examples).copy()
    selected.insert(0, "selection_rank", range(1, len(selected) + 1))
    selected["selection_rule"] = "error_count desc, error_confidence desc, sample_id asc"
    manifest_path = artifact_dir / "failure_gallery_manifest.csv"
    selected.to_csv(manifest_path, index=False)

    columns = 4
    rows = max(1, int(np.ceil(max(len(selected), 1) / columns)))
    fig, axes = plt.subplots(rows, columns, figsize=(12, 3.15 * rows), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    if selected.empty:
        axes[0, 0].text(0.5, 0.5, "No test errors", ha="center", va="center", fontsize=16)
    else:
        for axis, (_, failure) in zip(axes.flat, selected.iterrows()):
            sample_id = str(failure.sample_id)
            image = None
            if sample_id in lookup.index:
                rgb_path = lookup.loc[sample_id].get("rgb_path", "")
                if isinstance(rgb_path, str) and rgb_path and Path(rgb_path).exists():
                    image = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
            if image is None:
                image = np.full((224, 224, 3), 235, dtype=np.uint8)
                axis.text(0.5, 0.5, "No RGB input", ha="center", va="center")
            else:
                axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axis.set_title(
                f"{sample_id} | {failure.source}\n"
                f"D {int(failure.disaster_gt)}→{int(failure.disaster_pred)}; "
                f"V {int(failure.victim_gt)}→{int(failure.victim_pred)}",
                fontsize=8,
            )
            axis.axis("off")
    fig.suptitle(
        "Deterministic test failures: both-task errors first, then confidence, then sample ID",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    gallery_path = artifact_dir / "failure_gallery.png"
    fig.savefig(gallery_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return gallery_path, manifest_path


def finalize_run_artifacts(
    context: dict,
    predictions: Iterable[dict],
    test_metrics: dict,
    output_dir: Path,
    exp_name: str,
    device: torch.device,
) -> dict:
    artifact_dir = context["artifact_dir"]
    frame = pd.DataFrame(list(predictions))
    predictions_path = artifact_dir / "test_predictions.csv"
    frame.to_csv(predictions_path, index=False)
    per_class, per_source = save_detailed_metrics(frame, artifact_dir)
    gallery, failure_manifest = save_failure_gallery(frame, context["metadata_dir"], artifact_dir)

    history_path = output_dir / "logs" / f"{exp_name}_history.json"
    if history_path.exists():
        history = pd.DataFrame(json.loads(history_path.read_text(encoding="utf-8")))
        history.to_csv(artifact_dir / "epoch_metrics.csv", index=False)

    best = output_dir / "checkpoints" / f"{exp_name}_best.pth"
    final = output_dir / "checkpoints" / f"{exp_name}_final.pth"
    peak_vram = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    peak_vram_reserved = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
    summary = {
        "experiment": exp_name,
        "seed": context["seed"],
        "wall_clock_seconds": time.time() - context["started_unix"],
        "peak_vram_bytes": peak_vram,
        "peak_vram_reserved_bytes": peak_vram_reserved,
        "test_metrics": {key: float(value) for key, value in test_metrics.items()},
        "checkpoints": {
            "best": str(best) if best.exists() else None,
            "best_sha256": sha256_file(best) if best.exists() else None,
            "final": str(final) if final.exists() else None,
            "final_sha256": sha256_file(final) if final.exists() else None,
        },
        "artifacts": {
            "predictions": str(predictions_path),
            "per_class_metrics": str(per_class),
            "per_source_metrics": str(per_source),
            "failure_gallery": str(gallery),
            "failure_manifest": str(failure_manifest),
        },
    }
    (artifact_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
