"""Precompute the exact normalized 64x63 log-mel tensors used by the loader.

This is preprocessing only; it does not train or evaluate a model. The cache key
is the SHA-1 of the resolved source path, allowing split-specific audio pairing
without duplicating feature files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.multimodal_dataset import load_audio_logmel, normalize_logmel, pad_or_crop_logmel


def cache_key(path: str) -> str:
    return hashlib.sha1(str(Path(path).resolve()).encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache AdapFuse audio log-mel features")
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=ROOT / "data" / "metadata_grouped" / "seed_42",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "audio_cache" / "logmel_64x63",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    frames = [pd.read_csv(args.metadata_dir / f"{split}.csv") for split in ("train", "val", "test")]
    audio_paths = sorted(
        set(pd.concat(frames, ignore_index=True)["audio_path"].dropna().astype(str)) - {""}
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = skipped = failed = 0
    failures = []

    for source in tqdm(audio_paths, desc="Caching audio"):
        destination = args.output_dir / f"{cache_key(source)}.npy"
        if destination.exists() and not args.overwrite:
            skipped += 1
            continue
        if not Path(source).exists():
            failed += 1
            failures.append({"path": source, "reason": "source missing"})
            continue
        try:
            feature = load_audio_logmel(source, n_mels=64)
            feature = pad_or_crop_logmel(normalize_logmel(feature), 63).astype(np.float32)
            np.save(destination, feature, allow_pickle=False)
            written += 1
        except Exception as exc:
            failed += 1
            failures.append({"path": source, "reason": repr(exc)})

    manifest = {
        "metadata_dir": str(args.metadata_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "unique_audio_paths": len(audio_paths),
        "written": written,
        "skipped": skipped,
        "failed": failed,
        "shape": [1, 64, 63],
        "failures": failures,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
