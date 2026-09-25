"""Create a non-destructive archive index of completed object-detection runs.

The source ``training_summary.json`` files and checkpoints are never modified.
Legacy D-Fire runs used the reversed display names (fire, smoke), although the
dataset's official class ids are 0=smoke and 1=fire.  The generated report table
normalizes those two per-class columns and records that correction explicitly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    return default if value is None else value


def find_best_weights(summary_path: Path) -> Path | None:
    candidate = summary_path.parent / "weights" / "best.pt"
    return candidate if candidate.is_file() else None


def report_row(root: Path, summary_path: Path) -> dict[str, Any]:
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    config = summary.get("config", {})
    metrics = summary.get("test_metrics") or {}
    configured_names = nested(config, "dataset", "class_names", default=[])
    legacy_reversed = configured_names == ["fire", "smoke"]

    # Metric keys were generated from the configured names.  Remap only the
    # normalized table columns; keep the source JSON byte-for-byte unchanged.
    if legacy_reversed:
        smoke_ap50 = metrics.get("AP50_fire")
        fire_ap50 = metrics.get("AP50_smoke")
    else:
        smoke_ap50 = metrics.get("AP50_smoke")
        fire_ap50 = metrics.get("AP50_fire")

    best = find_best_weights(summary_path)
    architecture = (
        summary.get("architecture")
        or nested(config, "model", "architecture")
        or metrics.get("architecture")
    )
    status = summary.get("status")
    if not status:
        status = "legacy_summary_without_metrics" if not metrics else "completed"

    return {
        "experiment_name": summary.get("experiment_name", summary_path.parent.name),
        "status": status,
        "architecture": architecture,
        "epochs_configured": nested(config, "training", "epochs"),
        "training_time_hours": summary.get("total_training_time_hours"),
        "test_precision": metrics.get("precision"),
        "test_recall": metrics.get("recall"),
        "test_mAP50": metrics.get("mAP50"),
        "test_mAP50_95": metrics.get("mAP50_95"),
        "test_AP50_smoke": smoke_ap50,
        "test_AP50_fire": fire_ap50,
        "official_class_mapping": "0=smoke;1=fire",
        "legacy_class_names_corrected": legacy_reversed,
        "timestamp": summary.get("timestamp"),
        "summary_path": summary_path.relative_to(root).as_posix(),
        "summary_sha256": sha256(summary_path),
        "best_weights_path": best.relative_to(root).as_posix() if best else None,
        "best_weights_sha256": sha256(best) if best else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = (args.output_dir or root / "outputs").resolve()
    summary_paths = sorted(
        path
        for path in output_dir.rglob("training_summary.json")
        if "_smoke_runs" not in path.parts
    )
    rows = [report_row(root, path) for path in summary_paths]

    manifest_path = output_dir / "experiment_results_manifest.json"
    table_path = output_dir / "experiment_results_table.csv"
    manifest = {
        "schema_version": 1,
        "purpose": "Report-ready index; original summaries and checkpoints are unchanged.",
        "official_class_mapping": {"0": "smoke", "1": "fire"},
        "excluded_directory_names": ["_smoke_runs"],
        "run_count": len(rows),
        "runs": rows,
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    fieldnames = list(rows[0]) if rows else []
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)

    print(f"Indexed {len(rows)} non-smoke-test runs")
    print(manifest_path)
    print(table_path)


if __name__ == "__main__":
    main()
