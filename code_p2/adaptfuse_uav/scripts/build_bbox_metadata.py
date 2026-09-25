"""
build_bbox_metadata.py
Scans SARD and C2A label directories, matches label files to CSV rows by image basename,
adds bbox_path and has_bbox columns to all metadata CSVs.
Run once from project root:
    python scripts/build_bbox_metadata.py
"""
import os
import sys
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

META_DIR  = ROOT / "data" / "metadata"
SARD_BASE = ROOT / "datasets" / "raw" / "SARD"
C2A_BASE  = ROOT / "datasets" / "raw" / "C2A" / "C2A_Dataset" / "new_dataset3"

# ── 1. Build lookup: image_stem -> label_path ──────────────────────────────────
print("[*] Scanning SARD label files...")
sard_map = {}
for labels_dir in SARD_BASE.rglob("labels"):
    for lf in labels_dir.glob("*.txt"):
        if "README" in lf.name:
            continue
        sard_map[lf.stem] = str(lf)
print(f"    Found {len(sard_map)} SARD label files.")

print("[*] Scanning C2A label files...")
c2a_map = {}
for split in ["train", "val", "test"]:
    labels_dir = C2A_BASE / split / "labels"
    if not labels_dir.exists():
        continue
    for lf in labels_dir.glob("*.txt"):
        c2a_map[lf.stem] = str(lf)
print(f"    Found {len(c2a_map)} C2A label files.")

combined_map = {**sard_map, **c2a_map}
print(f"[*] Total unique label stems: {len(combined_map)}")

# ── 2. Apply to each CSV ───────────────────────────────────────────────────────
csvs = list(META_DIR.glob("*.csv"))
print(f"\n[*] Processing {len(csvs)} CSV files in {META_DIR}...")

def find_bbox(row):
    rgb = str(row.get("rgb_path", ""))
    if not rgb or rgb == "nan":
        return "", 0
    stem = Path(rgb).stem
    lp = combined_map.get(stem, "")
    return lp, (1 if lp else 0)

for csv_path in sorted(csvs):
    df = pd.read_csv(csv_path)
    results = df.apply(find_bbox, axis=1, result_type="expand")
    df["bbox_path"] = results[0]
    df["has_bbox"]  = results[1]
    df.to_csv(csv_path, index=False)
    n_bbox = df["has_bbox"].sum()
    print(f"  {csv_path.name}: {len(df)} rows | {n_bbox} with bbox ({100*n_bbox/max(len(df),1):.1f}%)")

print("\n[DONE] All CSVs updated with bbox_path and has_bbox columns.")
