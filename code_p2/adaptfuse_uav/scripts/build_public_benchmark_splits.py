"""
Build metadata splits that follow the published protocols of two public datasets,
so pipeline-trained models can be compared with published results.

  FLAME (Shamsoshoara et al., 2021): official Training folder (39,375 frames,
    Matrice 200 / Zenmuse X4S) split 80/20 into train/val as in the paper;
    official Test folder (8,617 frames, Phantom 3) as test. Binary: 0=non-fire, 1=fire.

  AIDER subset (6,433 images): per-class train/val/test counts from Table I of
    Lee et al. (WATT-EffNet, 2023), which follows the 4:1:2 AIDER protocol.
    Five classes: 0=normal, 1=fire, 2=collapsed_building, 3=flooded_areas,
    4=traffic_incident.

Rows reuse the paths (RGB + RGB-derived pseudo-thermal) from all_samples.csv.
Usage: python scripts/build_public_benchmark_splits.py
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent.parent
META = ROOT / "data" / "metadata"
OUT = ROOT / "data" / "metadata_public"
SEED = 42


def _prep(df, split):
    df = df.copy()
    df["victim_label"] = 0
    df["has_audio"] = 0
    df["audio_path"] = ""
    df["degradation"] = "none"
    df["split"] = split
    return df


def _write(name, train, val, test):
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    for split, df in (("train", train), ("val", val), ("test", test)):
        _prep(df, split).to_csv(d / f"{split}.csv", index=False)
    print(f"{name}: train={len(train)} val={len(val)} test={len(test)}")
    for split, df in (("train", train), ("val", val), ("test", test)):
        print("  ", split, df["disaster_label"].value_counts().sort_index().to_dict())


def build_flame(allr):
    f = allr[allr.source_dataset == "flame"].copy()
    folder = f.rgb_path.str.replace("\\", "/", regex=False)
    f["disaster_label"] = folder.str.contains("/fire/").astype(int)
    official_train = f[folder.str.contains("/FLAME/Training/")]
    official_test = f[folder.str.contains("/FLAME/Test/")]
    val = official_train.groupby("disaster_label", group_keys=False).sample(frac=0.2, random_state=SEED)
    train = official_train.drop(val.index)
    assert len(official_train) == 39375 and len(official_test) == 8617
    _write("flame_official", train, val, official_test)


AIDER_CLASSES = {  # folder: (label, n_train, n_val, n_test) from WATT-EffNet Table I
    "normal": (0, 2107, 527, 1756),
    "fire": (1, 249, 63, 209),
    "collapsed_building": (2, 367, 41, 103),
    "flooded_areas": (3, 252, 63, 211),
    "traffic_incident": (4, 232, 59, 194),
}


def build_aider(allr):
    a = allr[allr.source_dataset == "aider"].copy()
    folder = a.rgb_path.str.replace("\\", "/", regex=False).str.extract(r"/AIDER/([^/]+)/")[0]
    parts = {"train": [], "val": [], "test": []}
    for cls, (label, n_tr, n_va, n_te) in AIDER_CLASSES.items():
        rows = a[folder == cls].sample(frac=1.0, random_state=SEED).copy()
        assert len(rows) == n_tr + n_va + n_te, (cls, len(rows))
        rows["disaster_label"] = label
        parts["train"].append(rows.iloc[:n_tr])
        parts["val"].append(rows.iloc[n_tr:n_tr + n_va])
        parts["test"].append(rows.iloc[n_tr + n_va:])
    _write("aider_412", *(pd.concat(parts[s]) for s in ("train", "val", "test")))


def main():
    allr = pd.read_csv(META / "all_samples.csv")
    build_flame(allr)
    build_aider(allr)


if __name__ == "__main__":
    main()
