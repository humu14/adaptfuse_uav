"""Build a deterministic CLAHE-enhanced copy of D-Fire.

The original dataset remains untouched. All three splits receive identical
preprocessing, labels are copied in official D-Fire order (0=smoke, 1=fire),
and any boundary-crossing YOLO boxes are clipped to the image canvas.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def enhance_image(source: Path, target: Path, clip_limit: float, grid_size: int) -> None:
    """Apply CLAHE to luminance while preserving chroma."""
    if target.exists() and target.stat().st_size > 0:
        return
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {source}")
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    enhanced = cv2.cvtColor(
        cv2.merge((clahe.apply(lightness), a_channel, b_channel)),
        cv2.COLOR_LAB2BGR,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.stem}.partial{target.suffix}")
    params = [cv2.IMWRITE_JPEG_QUALITY, 95] if target.suffix.lower() in {".jpg", ".jpeg"} else []
    if not cv2.imwrite(str(temporary), enhanced, params):
        raise RuntimeError(f"Could not write image: {temporary}")
    os.replace(temporary, target)


def clipped_label(text: str) -> tuple[str, int, int]:
    """Clip normalized xywh boxes and drop geometrically empty boxes."""
    output = []
    corrections = 0
    removed = 0
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"Expected five fields on label line {line_number}: {line!r}")
        class_id = int(float(fields[0]))
        if class_id not in {0, 1}:
            raise ValueError(f"Unexpected D-Fire class ID {class_id} on line {line_number}")
        x, y, width, height = map(float, fields[1:])
        x1, y1 = max(0.0, x - width / 2), max(0.0, y - height / 2)
        x2, y2 = min(1.0, x + width / 2), min(1.0, y + height / 2)
        if x2 <= x1 or y2 <= y1:
            removed += 1
            continue
        new_x, new_y = (x1 + x2) / 2, (y1 + y2) / 2
        new_width, new_height = x2 - x1, y2 - y1
        new_values = (new_x, new_y, new_width, new_height)
        if any(abs(old - new) > 1e-9 for old, new in zip((x, y, width, height), new_values)):
            corrections += 1
        output.append(
            f"{class_id} {new_x:.10g} {new_y:.10g} {new_width:.10g} {new_height:.10g}"
        )
    return ("\n".join(output) + ("\n" if output else ""), corrections, removed)


def process_split(source_root: Path, output_root: Path, split: str, args) -> dict:
    images = sorted(
        path for path in (source_root / "images" / split).iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    source_labels = source_root / "labels" / split
    output_images = output_root / "images" / split
    output_labels = output_root / "labels" / split
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    def process_image(path: Path) -> None:
        enhance_image(path, output_images / path.name, args.clip_limit, args.grid_size)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, _ in enumerate(executor.map(process_image, images), start=1):
            if index % 500 == 0 or index == len(images):
                print(f"{split}: enhanced {index}/{len(images)} images", flush=True)

    corrections = 0
    removed = 0
    positive = 0
    boxes = [0, 0]
    for image in images:
        source_label = source_labels / f"{image.stem}.txt"
        if not source_label.exists():
            raise FileNotFoundError(f"Missing label for {image}: {source_label}")
        fixed, changed, dropped = clipped_label(source_label.read_text(encoding="utf-8"))
        corrections += changed
        removed += dropped
        if fixed:
            positive += 1
            for line in fixed.splitlines():
                boxes[int(line.split()[0])] += 1
        (output_labels / source_label.name).write_text(fixed, encoding="utf-8")
    return {
        "images": len(images),
        "positive_images": positive,
        "negative_images": len(images) - positive,
        "boxes": {"smoke": boxes[0], "fire": boxes[1]},
        "clipped_boxes": corrections,
        "removed_degenerate_boxes": removed,
    }


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=project_root / "data/datasets/dfire")
    parser.add_argument("--output", type=Path, default=project_root / "data/datasets/dfire_clahe")
    parser.add_argument("--clip-limit", type=float, default=2.0)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.source, args.output = args.source.resolve(), args.output.resolve()
    if args.source == args.output:
        raise ValueError("Source and output directories must be different")

    report = {
        "source": str(args.source),
        "output": str(args.output),
        "preprocessing": "CLAHE on LAB luminance",
        "clip_limit": args.clip_limit,
        "grid_size": args.grid_size,
        "class_names": {"0": "smoke", "1": "fire"},
        "splits": {},
    }
    for split in ("train", "val", "test"):
        report["splits"][split] = process_split(args.source, args.output, split, args)

    dataset_yaml = f"""path: {args.output.as_posix()}
train: images/train
val: images/val
test: images/test
names:
  0: smoke
  1: fire
nc: 2
"""
    (args.output / "dataset.yaml").write_text(dataset_yaml, encoding="utf-8")
    (args.output / "preparation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
