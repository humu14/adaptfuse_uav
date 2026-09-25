"""
Model Comparison — Side-by-side benchmark results
===================================================
Reads training results from all benchmark models and generates
a comparison table, ranked by mAP@50.

Usage:
    python evaluation/compare_models.py --results_dir outputs/
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.build_model import get_model_info


def collect_results(results_dir: str) -> list:
    """Collect results from all trained models in the results directory."""
    results_dir = Path(results_dir)
    all_results = []

    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if run_dir.name.startswith("_") or run_dir.name.startswith("smoke_"):
            continue

        # Look for training_summary.json or results from Ultralytics
        summary_path = run_dir / "training_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            metrics = summary.get("test_metrics") or summary.get("validation_metrics") or {}
            summary.update(metrics)
            all_results.append(summary)
            continue

        # Check for Ultralytics-style results
        for sub in run_dir.iterdir():
            if sub.is_dir():
                results_csv = sub / "results.csv"
                if results_csv.exists():
                    result = _parse_ultralytics_results(sub, run_dir.name)
                    if result:
                        all_results.append(result)

    return all_results


def _parse_ultralytics_results(run_dir: Path, model_name: str) -> dict:
    """Parse Ultralytics training results from results.csv."""
    import csv

    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        return None

    # Read the last row of results.csv for final metrics
    with open(results_csv) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return None

    last = rows[-1]

    # Ultralytics column names vary by version, try common ones
    result = {
        "architecture": model_name,
        "status": "completed",
        "epochs_completed": len(rows),
    }

    # Try to extract mAP metrics
    for key in last:
        key_clean = key.strip()
        val = last[key].strip() if isinstance(last[key], str) else last[key]
        try:
            val = float(val)
        except (ValueError, TypeError):
            continue

        if "map50-95" in key_clean.lower() or "map_0.5:0.95" in key_clean.lower():
            result["mAP50_95"] = val
        elif "map50" in key_clean.lower() or "map_0.5" in key_clean.lower():
            result["mAP50"] = val
        elif "precision" in key_clean.lower():
            result["precision"] = val
        elif "recall" in key_clean.lower():
            result["recall"] = val
        elif "box_loss" in key_clean.lower() or "train/box_loss" in key_clean.lower():
            result["final_box_loss"] = val

    return result


def generate_comparison_table(results: list, output_path: str = None):
    """Generate and print a comparison table."""
    if not results:
        print("No results found!")
        return

    # COCO-style mAP@50:95 is the primary selection metric.
    results.sort(key=lambda r: r.get("mAP50_95", 0), reverse=True)

    # Print table
    print(f"\n{'='*90}")
    print(f"  AdapFuse-OD -- Benchmark Model Comparison")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Models: {len(results)}")
    print(f"{'='*90}\n")

    header = (f"  {'Rank':<5} {'Model':<28} {'Family':<10} {'Params':<8} "
              f"{'mAP@50':<9} {'mAP@50:95':<10} {'P':<7} {'R':<7} {'Status':<10}")
    print(header)
    print(f"  {'-'*84}")

    for rank, r in enumerate(results, 1):
        arch = r.get("architecture", "unknown")
        label = r.get("experiment_name", arch)
        info = get_model_info(arch)

        map50 = r.get("mAP50", 0)
        map50_95 = r.get("mAP50_95", 0)
        precision = r.get("precision", 0)
        recall = r.get("recall", 0)
        status = r.get("status", "unknown")

        # Highlight best model
        marker = " *" if rank == 1 else "  "

        print(f"{marker}{rank:<4} {label:<28} {info['family']:<10} {info['params_m']:<7.1f}M "
              f"{map50:<9.4f} {map50_95:<10.4f} {precision:<7.4f} {recall:<7.4f} {status:<10}")

    print(f"\n  {'-'*84}")
    print(f"  * = Best model (highest mAP@50:95)")
    print()

    # Per-class AP if available
    has_per_class = any("AP50_fire" in r for r in results)
    if has_per_class:
        print(f"  Per-Class AP@50:")
        print(f"  {'Model':<28} {'Fire':<10} {'Smoke':<10}")
        print(f"  {'-'*48}")
        for r in results:
            arch = r.get("architecture", "unknown")
            fire_ap = r.get("AP50_fire", 0)
            smoke_ap = r.get("AP50_smoke", 0)
            print(f"  {arch:<28} {fire_ap:<10.4f} {smoke_ap:<10.4f}")
        print()

    # Save to file if requested
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "timestamp": datetime.now().isoformat(),
            "num_models": len(results),
            "ranked_results": results,
            "best_model": results[0].get("experiment_name", results[0]["architecture"]) if results else None,
            "selection_metric": "mAP50_95",
        }
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"  Results saved to: {output_path}")

    # Print recommendation
    if results:
        best = results[0]
        print(f"\n  +-------------------------------------------------------+")
        print(f"  |  RECOMMENDATION                                       |")
        print(f"  |  Best model: {best['architecture']:<40} |")
        print(f"  |  mAP@50:95: {best.get('mAP50_95', 0):.4f}                                 |")
        print(f"  |                                                       |")
        print(f"  |  Next: report baseline-vs-HPG effect and latency       |")
        print(f"  +-------------------------------------------------------+\n")


def main():
    parser = argparse.ArgumentParser(description="Compare benchmark models")
    parser.add_argument("--results_dir", type=str, default="outputs",
                        help="Directory containing training results")
    parser.add_argument("--output", type=str, default=None,
                        help="Save comparison JSON to this path")
    args = parser.parse_args()

    results = collect_results(args.results_dir)
    generate_comparison_table(results, args.output)


if __name__ == "__main__":
    main()
