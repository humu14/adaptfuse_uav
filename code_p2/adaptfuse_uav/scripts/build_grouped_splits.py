"""Build leakage-resistant grouped train/validation/test metadata.

The original project performs a row-level stratified split. This script creates a
new, non-destructive split in ``data/metadata_grouped/seed_<N>`` and records the
grouping assumptions in ``split_report.json``.

Grouping rules
--------------
* FLAME: adjacent numbered frames are kept in fixed-size sequence blocks. The
  public files do not expose true video/flight identifiers, so this is an
  explicit proxy rather than a claim of flight-level separation.
* C2A: augmented variants sharing ``<scene>_imageNNNN`` stay together.
* SARD: Roboflow variants sharing the original ``gssNNNN`` id stay together.
* AIDER and FireNet: one source image per group (no sequence id is available).
* ESC-50: official folds 1--3/4/5 map to train/val/test. Visual rows with audio
  are re-paired only from the corresponding split's ESC-50 pool so no clip is
  reused across partitions.

Usage:
    python scripts/build_grouped_splits.py --seed 42 --flame-block-size 250
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "metadata"
DEFAULT_OUTPUT = ROOT / "data" / "metadata_grouped"
SPLITS = ("train", "val", "test")
FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}


def _stem(path_value: object) -> str:
    if pd.isna(path_value) or not str(path_value).strip():
        return "missing"
    return Path(str(path_value)).stem


def _flame_group(path_value: object, block_size: int) -> str:
    path = Path(str(path_value))
    stem = path.stem.lower()
    match = re.match(r"(.*?)(\d+)$", stem)
    if not match:
        return f"flame:{path.parent.name.lower()}:{stem}"
    prefix, number = match.group(1), int(match.group(2))
    parts_lower = [p.lower() for p in path.parts]
    try:
        flame_idx = parts_lower.index("flame")
        provenance = ":".join(parts_lower[flame_idx + 1 : -1])
    except ValueError:
        provenance = path.parent.name.lower()
    return f"flame:{provenance}:{prefix}:block_{number // block_size:05d}"


def capture_group(row: pd.Series, flame_block_size: int) -> str:
    source = str(row["source_dataset"]).lower()
    stem = _stem(row.get("rgb_path"))
    if source == "flame":
        return _flame_group(row.get("rgb_path"), flame_block_size)
    if source == "c2a":
        base = re.sub(r"_\d+$", "", stem.lower())
        return f"c2a:{base}"
    if source == "sard":
        match = re.match(r"(gss\d+)", stem.lower())
        return f"sard:{match.group(1) if match else stem.lower()}"
    if source in {"aider", "firenet"}:
        return f"{source}:{stem.lower()}"
    return f"{source}:{row['sample_id']}"


def esc_fold(path_value: object) -> int:
    match = re.match(r"([1-5])-", Path(str(path_value)).name)
    if not match:
        raise ValueError(f"Cannot read ESC-50 fold from {path_value}")
    return int(match.group(1))


def assign_groups(frame: pd.DataFrame, seed: int) -> pd.Series:
    """Greedily balance complete groups within source/label strata."""
    assignments: dict[str, str] = {}
    rng = np.random.default_rng(seed)
    strata = ["source_dataset", "disaster_label", "victim_label"]

    for _, stratum in frame.groupby(strata, dropna=False, sort=True):
        sizes = stratum.groupby("capture_group").size().rename("n").reset_index()
        sizes["tie"] = rng.random(len(sizes))
        sizes = sizes.sort_values(["n", "tie"], ascending=[False, True])
        targets = {name: FRACTIONS[name] * len(stratum) for name in SPLITS}
        counts = {name: 0 for name in SPLITS}

        for group_id, n_rows, _ in sizes.itertuples(index=False, name=None):
            candidate_scores = {}
            for split in SPLITS:
                proposed = dict(counts)
                proposed[split] += int(n_rows)
                candidate_scores[split] = sum(
                    ((proposed[name] - targets[name]) / max(targets[name], 1.0)) ** 2
                    for name in SPLITS
                )
            best_score = min(candidate_scores.values())
            tied = [name for name, score in candidate_scores.items() if math.isclose(score, best_score)]
            chosen = tied[int(rng.integers(0, len(tied)))]
            assignments[str(group_id)] = chosen
            counts[chosen] += int(n_rows)

    return frame["capture_group"].map(assignments)


def reassign_audio(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, int]:
    """Keep ESC clips partition-disjoint and deterministically re-pair visual rows."""
    result = frame.copy()
    rng = np.random.default_rng(seed + 10_000)
    is_esc = result["source_dataset"].str.lower().eq("esc50")
    missing_pool = 0

    pools: dict[tuple[str, int], list[str]] = {}
    for (split, label), group in result[is_esc].groupby(["split", "disaster_label"]):
        pools[(str(split), int(label))] = sorted(group["audio_path"].dropna().astype(str).unique())

    visual_audio = (~is_esc) & result["has_audio"].fillna(0).astype(int).eq(1)
    for (split, label), indices in result[visual_audio].groupby(["split", "disaster_label"]).groups.items():
        pool = list(pools.get((str(split), int(label)), []))
        if not pool:
            result.loc[list(indices), "audio_path"] = ""
            result.loc[list(indices), "has_audio"] = 0
            missing_pool += len(indices)
            continue
        rng.shuffle(pool)
        ordered_indices = np.array(list(indices), dtype=int)
        rng.shuffle(ordered_indices)
        assigned = [pool[i % len(pool)] for i in range(len(ordered_indices))]
        result.loc[ordered_indices, "audio_path"] = assigned

    return result, missing_pool


def overlap_count(frame: pd.DataFrame, column: str) -> dict[str, int]:
    values = {
        split: set(frame.loc[frame["split"].eq(split), column].dropna().astype(str)) - {""}
        for split in SPLITS
    }
    return {
        "train_val": len(values["train"] & values["val"]),
        "train_test": len(values["train"] & values["test"]),
        "val_test": len(values["val"] & values["test"]),
    }


def nested_counts(frame: pd.DataFrame, columns: list[str]) -> dict:
    table = frame.groupby(columns).size()
    output: dict = {}
    for keys, count in table.items():
        if not isinstance(keys, tuple):
            keys = (keys,)
        cursor = output
        for key in keys[:-1]:
            cursor = cursor.setdefault(str(key), {})
        cursor[str(keys[-1])] = int(count)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Create grouped, leakage-resistant metadata splits")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--flame-block-size", type=int, default=250)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_root / f"seed_{args.seed}"
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} is not empty; pass --overwrite to replace generated files")
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = [pd.read_csv(args.input_dir / f"{split}.csv") for split in SPLITS]
    data = pd.concat(frames, ignore_index=True)
    if data["sample_id"].duplicated().any():
        duplicates = int(data["sample_id"].duplicated().sum())
        raise ValueError(f"Input metadata contains {duplicates} duplicate sample_id values")

    data["original_split"] = data["split"].astype(str)
    is_esc = data["source_dataset"].str.lower().eq("esc50")
    visual = data[~is_esc].copy()
    audio_only = data[is_esc].copy()

    visual["capture_group"] = visual.apply(
        capture_group, axis=1, flame_block_size=args.flame_block_size
    )
    visual["split"] = assign_groups(visual, args.seed)

    audio_only["capture_group"] = audio_only["audio_path"].map(lambda p: f"esc50:{_stem(p).lower()}")
    folds = audio_only["audio_path"].map(esc_fold)
    audio_only["split"] = np.where(folds.le(3), "train", np.where(folds.eq(4), "val", "test"))

    grouped = pd.concat([visual, audio_only], ignore_index=True)
    grouped, missing_pool = reassign_audio(grouped, args.seed)
    grouped["split_seed"] = args.seed
    grouped = grouped.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    capture_overlap = overlap_count(grouped, "capture_group")
    audio_overlap = overlap_count(grouped[grouped["has_audio"].fillna(0).astype(int).eq(1)], "audio_path")
    if any(capture_overlap.values()) or any(audio_overlap.values()):
        raise AssertionError(f"Split leakage detected: capture={capture_overlap}, audio={audio_overlap}")

    for split in SPLITS:
        part = grouped[grouped["split"].eq(split)].copy()
        part.to_csv(output_dir / f"{split}.csv", index=False)

    report = {
        "seed": args.seed,
        "flame_block_size": args.flame_block_size,
        "fractions_requested": FRACTIONS,
        "rows_total": int(len(grouped)),
        "rows_by_split": grouped["split"].value_counts().sort_index().astype(int).to_dict(),
        "capture_groups_by_split": grouped.groupby("split")["capture_group"].nunique().astype(int).to_dict(),
        "rows_by_split_and_class": nested_counts(grouped, ["split", "disaster_label"]),
        "rows_by_split_and_source": nested_counts(grouped, ["split", "source_dataset"]),
        "capture_group_overlap": capture_overlap,
        "audio_path_overlap": audio_overlap,
        "visual_audio_rows_dropped_for_missing_split_pool": int(missing_pool),
        "grouping_limitations": [
            "FLAME has no delivered video/flight identifier; fixed contiguous frame blocks are a proxy.",
            "AIDER and FireNet expose no capture-sequence identifier, so each image is its own group.",
            "Audio remains label-matched rather than synchronized, but exact ESC-50 clips are split-disjoint.",
        ],
    }
    (output_dir / "split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\n[OK] Grouped metadata written to {output_dir}")


if __name__ == "__main__":
    main()
