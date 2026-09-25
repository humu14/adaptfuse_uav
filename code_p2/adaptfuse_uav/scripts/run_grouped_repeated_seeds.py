"""Run any YAML configuration on one grouped split across multiple training seeds."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "adaptfuse_v1_grouped_gtx960.yaml"
DEFAULT_METADATA = ROOT / "data" / "metadata_grouped" / "seed_42"
DEFAULT_OUTPUT = ROOT / "outputs" / "grouped_repeated"
METRICS = ("disaster_f1", "victim_f1", "combined_f1", "disaster_acc", "victim_acc", "loss")
T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}


def summarize(seed_dirs: list[tuple[int, Path]], output_path: Path, exp_name: str) -> dict:
    rows = []
    for seed, directory in seed_dirs:
        result_path = directory / "logs" / f"{exp_name}_test_results.json"
        if not result_path.exists():
            raise FileNotFoundError(f"Missing result for seed {seed}: {result_path}")
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        rows.append({"seed": seed, **payload["test"]})

    n = len(rows)
    if n < 2:
        raise ValueError("At least two completed seeds are required for a confidence interval")
    critical = T95.get(n, 1.96)
    aggregate = {}
    for metric in METRICS:
        values = [float(row[metric]) for row in rows if metric in row]
        if len(values) != n:
            continue
        mean = statistics.fmean(values)
        std = statistics.stdev(values)
        half = critical * std / math.sqrt(n)
        aggregate[metric] = {
            "mean": mean,
            "std": std,
            "ci95_low": mean - half,
            "ci95_high": mean + half,
            "n": n,
        }
    result = {"runs": rows, "aggregate": aggregate}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a grouped experiment across repeated seeds")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43, 44, 45])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--audio-cache-dir", type=Path, default=None)
    parser.add_argument("--teacher-checkpoint", type=Path, default=None)
    parser.add_argument("--dry-run-batches", type=int, default=None)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    exp_name = config.get("experiment_name", args.config.stem)

    seed_dirs = [(seed, args.output_root / f"seed_{seed}") for seed in args.seeds]
    if not args.summarize_only:
        for seed, output_dir in seed_dirs:
            command = [
                sys.executable,
                str(ROOT / "training" / "train.py"),
                "--config", str(args.config),
                "--metadata_dir", str(args.metadata_dir),
                "--output_dir", str(output_dir),
                "--seed", str(seed),
            ]
            if args.epochs is not None:
                command += ["--epochs", str(args.epochs)]
            if args.batch_size is not None:
                command += ["--batch_size", str(args.batch_size)]
            if args.num_workers is not None:
                command += ["--num_workers", str(args.num_workers)]
            if args.audio_cache_dir is not None:
                command += ["--audio_cache_dir", str(args.audio_cache_dir)]
            if args.teacher_checkpoint is not None:
                command += ["--teacher_ckpt", str(args.teacher_checkpoint)]
            if args.dry_run_batches is not None:
                command += ["--dry_run_batches", str(args.dry_run_batches)]
            print(f"\n[RUN seed={seed}] {' '.join(command)}", flush=True)
            subprocess.run(command, cwd=ROOT, check=True)

    result = summarize(
        seed_dirs,
        args.output_root / f"{exp_name}_repeated_summary.json",
        exp_name,
    )
    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
