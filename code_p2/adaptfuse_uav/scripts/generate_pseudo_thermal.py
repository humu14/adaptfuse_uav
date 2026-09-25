"""
AdapFuse-UAV: Pseudo-Thermal Generator
=======================================
Converts all RGB images to pseudo-thermal using CLAHE-enhanced grayscale.

Strategy:
  1. Load RGB image
  2. Convert to grayscale
  3. Apply CLAHE (enhances local contrast → simulates thermal texture)
  4. Save grayscale PNG to datasets/pseudo_thermal/ mirroring datasets/raw/ structure

Output path mirrors input:
  datasets/raw/FLAME/Training/fire/img.jpg
  → datasets/pseudo_thermal/FLAME/Training/fire/img.png

Run:
  python scripts/generate_pseudo_thermal.py [--workers N] [--skip-existing]
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

ROOT       = Path(__file__).parent.parent
RAW_DIR    = ROOT / "datasets" / "raw"
THERM_DIR  = ROOT / "datasets" / "pseudo_thermal"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

CLAHE = None  # created per-process


def _init_clahe():
    global CLAHE
    CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


def convert_one(args):
    src, dst, skip_existing = args
    if skip_existing and dst.exists():
        return "skip"
    try:
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            return f"fail:{src}"
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        enhanced = CLAHE.apply(gray)
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), enhanced)
        return "ok"
    except Exception as e:
        return f"fail:{src}:{e}"


def gather_tasks(skip_existing: bool):
    tasks = []
    for src in RAW_DIR.rglob("*"):
        if src.is_file() and src.suffix.lower() in IMG_EXTS:
            rel = src.relative_to(RAW_DIR)
            dst = THERM_DIR / rel.with_suffix(".png")
            tasks.append((src, dst, skip_existing))
    return tasks


def main(workers: int, skip_existing: bool):
    print("=" * 62)
    print("AdapFuse-UAV: Pseudo-Thermal Generation (CLAHE grayscale)")
    print("=" * 62)

    tasks = gather_tasks(skip_existing)
    print(f"  Images found : {len(tasks):,}")
    if skip_existing:
        pending = [(s, d, se) for s, d, se in tasks if not d.exists()]
        print(f"  Already done : {len(tasks) - len(pending):,}")
        print(f"  To generate  : {len(pending):,}")
        tasks = pending
    else:
        print(f"  To generate  : {len(tasks):,}")

    if not tasks:
        print("  Nothing to do.")
        return

    ok = skip = fail = 0

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_clahe,
    ) as exe:
        futures = {exe.submit(convert_one, t): t for t in tasks}
        with tqdm(total=len(tasks), desc="Generating pseudo-thermal") as pbar:
            for fut in as_completed(futures):
                result = fut.result()
                if result == "ok":
                    ok += 1
                elif result == "skip":
                    skip += 1
                else:
                    fail += 1
                    tqdm.write(f"[FAIL] {result}")
                pbar.update(1)

    print(f"\n  Done — ok:{ok}  skipped:{skip}  failed:{fail}")
    print(f"  Saved to: {THERM_DIR.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workers", type=int,
        default=max(1, multiprocessing.cpu_count() - 1),
        help="Parallel worker processes",
    )
    parser.add_argument(
        "--skip-existing", action="store_true", default=True,
        help="Skip already-generated files (default: True)",
    )
    parser.add_argument(
        "--no-skip", dest="skip_existing", action="store_false",
        help="Regenerate all, overwriting existing files",
    )
    args = parser.parse_args()
    main(args.workers, args.skip_existing)
