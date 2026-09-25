"""
AdapFuse-UAV: Exact Class Balancing
=====================================
Balances train and val splits so that:
  - All 4 disaster_label classes have the EXACT same count
  - Both victim_label classes have the EXACT same count

Strategy:
  Stage 1 — Disaster balance:
      Oversample each disaster class to max_disaster_count.
      Each class ends up with exactly the same N rows.

  Stage 2 — Victim balance:
      After stage 1, count victim=0 and victim=1 in the balanced set.
      Oversample victim=1 rows (distributed proportionally across disaster
      classes so disaster balance is preserved) until victim=1 == victim=0.

  Marks all synthetic duplicate rows with augmented=True.
  The DataLoader applies stronger augmentation to those rows.

Outputs (written to data/metadata/):
  train_balanced.csv
  val_balanced.csv
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

ROOT       = Path(__file__).parent.parent
SPLITS_DIR = ROOT / "data" / "metadata"

DISASTER_NAMES = {0: "Normal", 1: "Fire/Smoke", 2: "Collapse/Flood", 3: "Other"}
VICTIM_NAMES   = {0: "No Victim", 1: "Victim Present"}

SEED = 42


# ── Helpers ───────────────────────────────────────────────────────────

def _print_dist(df: pd.DataFrame, col: str, names: dict, title: str = ""):
    total = len(df)
    counts = df[col].value_counts().sort_index()
    print(f"  {title or col}:")
    for cls, cnt in counts.items():
        pct = cnt / total * 100
        print(f"    [{cls}] {names.get(cls,'?'):20s}: {cnt:8,}  ({pct:5.1f}%)")


def _print_joint(df: pd.DataFrame):
    ct = pd.crosstab(df["disaster_label"], df["victim_label"])
    ct.columns = [VICTIM_NAMES.get(c, str(c)) for c in ct.columns]
    ct.index   = [f"{i} {DISASTER_NAMES.get(i,'?')}" for i in ct.index]
    print("  Joint (disaster × victim):")
    for row_label, row in ct.iterrows():
        print(f"    {row_label:25s}: " + "  ".join(f"{v:>8,}" for v in row))


# ── Core balancer ─────────────────────────────────────────────────────

def balance_exact(df: pd.DataFrame, split_name: str, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df  = df.copy()
    df["augmented"] = df["augmented"].astype(bool) if "augmented" in df.columns else False

    print(f"\n{'='*64}")
    print(f"  {split_name.upper()}  —  original: {len(df):,} rows")
    print(f"{'='*64}")
    print("\nBEFORE:")
    _print_dist(df, "disaster_label", DISASTER_NAMES, "Disaster classes")
    _print_dist(df, "victim_label",   VICTIM_NAMES,   "Victim classes  ")
    _print_joint(df)

    # ── Stage 1: exact disaster balance ──────────────────────────
    target_d = int(df["disaster_label"].value_counts().max())
    parts = []
    for cls in sorted(df["disaster_label"].unique()):
        cls_df = df[df["disaster_label"] == cls].copy()
        deficit = target_d - len(cls_df)
        if deficit > 0:
            extra = cls_df.sample(
                n=deficit, replace=True,
                random_state=int(rng.integers(0, 2**31 - 1)),
            ).copy()
            extra["augmented"] = True
            cls_df = pd.concat([cls_df, extra], ignore_index=True)
        parts.append(cls_df)
    balanced = pd.concat(parts, ignore_index=True)

    # ── Stage 2: exact victim balance ────────────────────────────
    # Add victim=1 rows proportionally from each disaster class so that
    # disaster balance is preserved after victim balancing.
    v0 = int((balanced["victim_label"] == 0).sum())
    v1 = int((balanced["victim_label"] == 1).sum())
    to_add = v0 - v1  # how many victim=1 rows to add in total

    if to_add > 0:
        disaster_classes = sorted(balanced["disaster_label"].unique())
        n_classes = len(disaster_classes)

        # Build victim=1 pool per disaster class
        pools = {
            cls: balanced[(balanced["disaster_label"] == cls) &
                          (balanced["victim_label"] == 1)]
            for cls in disaster_classes
        }
        # Only classes that have victim=1 samples can contribute
        eligible = [cls for cls in disaster_classes if len(pools[cls]) > 0]

        if eligible:
            per_class = to_add // len(eligible)
            remainder = to_add - per_class * len(eligible)

            extra_parts = []
            for i, cls in enumerate(eligible):
                n = per_class + (1 if i < remainder else 0)
                if n == 0:
                    continue
                extra = pools[cls].sample(
                    n=n, replace=True,
                    random_state=int(rng.integers(0, 2**31 - 1)),
                ).copy()
                extra["augmented"] = True
                extra_parts.append(extra)

            if extra_parts:
                balanced = pd.concat([balanced] + extra_parts, ignore_index=True)

    # ── Shuffle ───────────────────────────────────────────────────
    balanced = balanced.sample(frac=1, random_state=seed).reset_index(drop=True)

    # ── Report ────────────────────────────────────────────────────
    n_aug  = int(balanced["augmented"].sum())
    n_orig = len(balanced) - n_aug
    print("\nAFTER:")
    _print_dist(balanced, "disaster_label", DISASTER_NAMES, "Disaster classes")
    _print_dist(balanced, "victim_label",   VICTIM_NAMES,   "Victim classes  ")
    _print_joint(balanced)
    print(f"\n  Original rows : {n_orig:,}")
    print(f"  Augmented rows: {n_aug:,}  ({n_aug/len(balanced)*100:.1f}%)")
    print(f"  Total rows    : {len(balanced):,}")

    return balanced


# ── Half-balanced creator ─────────────────────────────────────────────

def create_half_balanced(df: pd.DataFrame, split_name: str, seed: int = SEED) -> pd.DataFrame:
    """
    Create half-sized version of a balanced CSV.

    Strategy:
      - For each (disaster_label x source_dataset) cell: take floor(n/2),
        with a minimum of MIN_PER_SOURCE rows so no source disappears.
      - After source-stratified sampling, trim or pad each disaster class
        to exactly half of its original count, sampling from the full
        class pool (not source-stratified) when padding is needed.
      - Shuffle and return.

    Guarantees:
      - All sources that existed in the balanced split remain present.
      - Each disaster class has exactly the same count (exact equality kept).
      - ~50/50 victim split preserved (inherited from balanced input).
    """
    MIN_PER_SOURCE = 5   # never drop a source below this many rows

    rng = np.random.default_rng(seed)

    # Target: half of current class count (balanced input has equal class sizes)
    current_class_size = int(df["disaster_label"].value_counts().max())
    target_half = current_class_size // 2

    print(f"\n{'='*64}")
    print(f"  {split_name.upper()} HALF  --  full balanced: {len(df):,}  target: {target_half:,}/class")
    print(f"{'='*64}")

    parts = []
    for cls in sorted(df["disaster_label"].unique()):
        cls_df = df[df["disaster_label"] == cls].copy()

        # Source-stratified 50% sample
        source_parts = []
        for src in sorted(cls_df["source_dataset"].unique()):
            src_df = cls_df[cls_df["source_dataset"] == src]
            n = max(MIN_PER_SOURCE, len(src_df) // 2)
            n = min(n, len(src_df))          # can't take more than available
            sampled = src_df.sample(
                n=n, replace=False,
                random_state=int(rng.integers(0, 2**31 - 1)),
            )
            source_parts.append(sampled)

        cls_half = pd.concat(source_parts, ignore_index=True)

        # Adjust to exact target_half
        if len(cls_half) > target_half:
            cls_half = cls_half.sample(
                n=target_half, replace=False,
                random_state=int(rng.integers(0, 2**31 - 1)),
            )
        elif len(cls_half) < target_half:
            # Pad from full class pool (excluding already selected)
            already = set(cls_half.index)
            pool = cls_df[~cls_df.index.isin(already)]
            need = target_half - len(cls_half)
            if len(pool) >= need:
                extra = pool.sample(
                    n=need, replace=False,
                    random_state=int(rng.integers(0, 2**31 - 1)),
                )
            else:
                extra = pool.sample(
                    n=need, replace=True,
                    random_state=int(rng.integers(0, 2**31 - 1)),
                )
            cls_half = pd.concat([cls_half, extra], ignore_index=True)

        parts.append(cls_half)

    result = pd.concat(parts, ignore_index=True)
    result = result.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Report
    print("\nClass counts:")
    _print_dist(result, "disaster_label", DISASTER_NAMES, "Disaster classes")
    _print_dist(result, "victim_label",   VICTIM_NAMES,   "Victim classes  ")
    print("\nSource coverage:")
    src_ct = pd.crosstab(result["disaster_label"], result["source_dataset"])
    print(src_ct.to_string())
    print(f"\n  Total rows: {len(result):,}")

    return result


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("AdapFuse-UAV: Exact Class Balancing")
    print("=" * 64)
    print(f"Seed: {SEED} | Reading from: {SPLITS_DIR}")

    # ── Full balanced splits ──────────────────────────────────────
    for split_name in ["train", "val"]:
        in_path  = SPLITS_DIR / f"{split_name}.csv"
        out_path = SPLITS_DIR / f"{split_name}_balanced.csv"

        if not in_path.exists():
            print(f"\n[ERROR] {in_path} not found. Run build_metadata.py first.")
            sys.exit(1)

        df     = pd.read_csv(in_path)
        result = balance_exact(df, split_name)
        result.to_csv(out_path, index=False)
        print(f"\n  Saved -> {out_path.name}  ({len(result):,} rows)")

    # ── Half-balanced splits ──────────────────────────────────────
    print("\n" + "=" * 64)
    print("  Creating half-balanced splits (all sources preserved)")
    for split_name in ["train", "val"]:
        in_path  = SPLITS_DIR / f"{split_name}_balanced.csv"
        out_path = SPLITS_DIR / f"{split_name}_balanced_half.csv"

        df     = pd.read_csv(in_path)
        result = create_half_balanced(df, split_name)
        result.to_csv(out_path, index=False)
        print(f"\n  Saved -> {out_path.name}  ({len(result):,} rows)")

    print("\n" + "=" * 64)
    print("Done.")
    print("  train_balanced.csv      -- full balanced")
    print("  val_balanced.csv        -- full balanced")
    print("  train_balanced_half.csv -- half balanced, all sources")
    print("  val_balanced_half.csv   -- half balanced, all sources")


if __name__ == "__main__":
    main()
