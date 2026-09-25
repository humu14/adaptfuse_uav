"""
D-Fire Dataset Setup & Verifier
===============================
Directly links the extracted 21,527 image D-Fire dataset from Kaggle cache
(`~/.cache/kagglehub/datasets/sayedgamal99/smoke-fire-detection-yolo/versions/1/data`)
to `data/datasets/dfire/dataset.yaml`.

Classes: 0=smoke, 1=fire
"""

import os
import sys
import argparse
from pathlib import Path
from collections import Counter

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "datasets" / "dfire"
KAGGLE_CACHE_DATA_DIR = Path(os.path.expanduser("~")) / ".cache" / "kagglehub" / "datasets" / "sayedgamal99" / "smoke-fire-detection-yolo" / "versions" / "1" / "data"


def get_dataset_root() -> Path:
    """Get path to extracted dataset root."""
    if KAGGLE_CACHE_DATA_DIR.exists() and (KAGGLE_CACHE_DATA_DIR / "train").exists():
        print(f"Found dataset in Kaggle cache: {KAGGLE_CACHE_DATA_DIR}")
        return KAGGLE_CACHE_DATA_DIR

    print("Downloading dataset via kagglehub...")
    import kagglehub
    path = Path(kagglehub.dataset_download("sayedgamal99/smoke-fire-detection-yolo"))
    data_path = path / "data" if (path / "data").exists() else path
    return data_path


def create_dataset_yaml(output_dir: Path, data_root: Path):
    """Create dataset.yaml pointing to the dataset root."""
    output_dir.mkdir(parents=True, exist_ok=True)

    yaml_content = f"""# D-Fire Dataset Configuration
# Fire and Smoke Detection Dataset (21,527 images)

path: {data_root.resolve()}
train: train/images
val: val/images
test: val/images

# Classes
names:
  0: smoke
  1: fire

# Number of classes
nc: 2
"""
    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"Created dataset.yaml: {yaml_path}")
    print(f"  Points to: {data_root.resolve()}")


def verify_dataset(data_root: Path) -> bool:
    """Verify dataset structure and count boxes."""
    print(f"\n{'='*60}")
    print(f"  D-Fire Dataset Verification")
    print(f"  Path: {data_root}")
    print(f"{'='*60}\n")

    all_ok = True

    for split in ["train", "val"]:
        images_dir = data_root / split / "images"
        labels_dir = data_root / split / "labels"

        if not images_dir.is_dir():
            print(f"  [X] {split}/images/ NOT FOUND at {images_dir}")
            all_ok = False
            continue

        image_files = set()
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            image_files.update(f.stem for f in images_dir.glob(f"*{ext}"))

        label_files = set()
        if labels_dir.is_dir():
            label_files = {f.stem for f in labels_dir.glob("*.txt")}

        matched = image_files & label_files

        class_counts = Counter()
        total_boxes = 0
        for stem in matched:
            lbl_file = labels_dir / f"{stem}.txt"
            with open(lbl_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls_id = int(parts[0])
                        class_counts[cls_id] += 1
                        total_boxes += 1

        print(f"  {split}:")
        print(f"    Images:        {len(image_files)}")
        print(f"    Labels:        {len(label_files)}")
        print(f"    Matched pairs: {len(matched)}")
        print(f"    Total boxes:   {total_boxes}")
        for cls_id in sorted(class_counts.keys()):
            cls_name = {0: "fire", 1: "smoke"}.get(cls_id, f"class_{cls_id}")
            print(f"      {cls_name}: {class_counts[cls_id]} boxes")
        print()

    print(f"\n{'='*60}")
    if all_ok:
        print("  [OK] Dataset verification PASSED")
    else:
        print("  [X] Dataset verification FAILED")
    print(f"{'='*60}\n")

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="D-Fire Dataset Setup")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    output_dir = Path(args.output)

    # Get data root
    data_root = get_dataset_root()

    # Create dataset.yaml pointing directly to data_root
    create_dataset_yaml(output_dir, data_root)

    # Verify
    verify_dataset(data_root)


if __name__ == "__main__":
    main()
