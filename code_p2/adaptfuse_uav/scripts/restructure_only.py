"""
Helper script to restructure the unzipped datasets so build_metadata.py can find them.
This script does NOT perform unzip or extraction, it only moves directories/files.
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "datasets" / "raw"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

def restructure_flame():
    flame_dir = RAW_DIR / "FLAME"
    flame_unzipped = flame_dir / "flame"
    
    if not flame_unzipped.exists():
        # Check if already restructured
        if (flame_dir / "Training" / "fire").exists():
            print("[SKIP] FLAME already restructured.")
            return
        else:
            print("[WARNING] FLAME unzipped folder not found at", flame_unzipped)
            return

    print("\n=== Restructuring FLAME ===")
    
    # Define mapping
    mapping = [
        (flame_unzipped / "Training" / "Training" / "Fire", flame_dir / "Training" / "fire"),
        (flame_unzipped / "Training" / "Training" / "No_Fire", flame_dir / "Training" / "non-fire"),
        (flame_unzipped / "Test" / "Test" / "Fire", flame_dir / "Test" / "fire"),
        (flame_unzipped / "Test" / "Test" / "No_Fire", flame_dir / "Test" / "non-fire"),
    ]
    
    for src, dst in mapping:
        if not src.exists():
            print(f"  [WARNING] Source not found: {src}")
            continue
        dst.mkdir(parents=True, exist_ok=True)
        files = [f for f in src.iterdir() if f.is_file()]
        for f in tqdm(files, desc=f"Moving to {dst.name}"):
            dest_file = dst / f.name
            if dest_file.exists():
                dest_file.unlink()
            shutil.move(str(f), str(dest_file))
            
    # Clean up the unzipped dir
    try:
        shutil.rmtree(flame_unzipped)
        print("  Cleaned up temp flame unzipped folder.")
    except Exception as e:
        print(f"  Warning: could not delete {flame_unzipped}: {e}")
        
    print("[DONE] FLAME restructured.")


def restructure_sard():
    sard_dir = RAW_DIR / "SARD"
    sard_images = sard_dir / "images"
    
    # Check if already restructured
    if sard_images.exists() and len(list(sard_images.glob("*"))) > 0:
        print("[SKIP] SARD already restructured.")
        return

    print("\n=== Restructuring SARD ===")
    sard_images.mkdir(parents=True, exist_ok=True)
    
    # Collect all image files
    img_files = []
    for split_dir in ["train", "test"]:
        src_path = sard_dir / split_dir
        if src_path.exists():
            for f in src_path.rglob("*"):
                if f.is_file() and f.suffix.lower() in IMG_EXTS:
                    img_files.append(f)
                    
    if not img_files:
        print("  [WARNING] No SARD image files found to move.")
        return

    for f in tqdm(img_files, desc="Moving SARD images"):
        dest = sard_images / f.name
        if dest.exists():
            dest.unlink()
        shutil.move(str(f), str(dest))
        
    # Clean up train and test folders
    for split_dir in ["train", "test"]:
        p = sard_dir / split_dir
        if p.exists():
            try:
                shutil.rmtree(p)
            except Exception as e:
                print(f"  Warning: could not delete {p}: {e}")
                
    print("[DONE] SARD restructured.")


def restructure_firenet():
    firenet_dir = RAW_DIR / "FireNet"
    fire_target = firenet_dir / "fire"
    normal_target = firenet_dir / "normal"
    smoke_target = firenet_dir / "smoke"
    
    # Check if already restructured
    if fire_target.exists() and (len(list(fire_target.glob("*"))) > 0):
        print("[SKIP] FireNet already restructured.")
        return

    print("\n=== Restructuring FireNet ===")
    fire_target.mkdir(parents=True, exist_ok=True)
    normal_target.mkdir(parents=True, exist_ok=True)
    smoke_target.mkdir(parents=True, exist_ok=True)
    
    # Collect all image files
    img_files = []
    for split_dir in ["train", "validation"]:
        src_path = firenet_dir / split_dir
        if src_path.exists():
            for f in src_path.rglob("*"):
                if f.is_file() and f.suffix.lower() in IMG_EXTS:
                    img_files.append(f)
                    
    if not img_files:
        print("  [WARNING] No FireNet image files found to move.")
        return

    for f in tqdm(img_files, desc="Moving FireNet images"):
        path_str = f.name.lower()
        if "smoke" in path_str:
            dest = smoke_target / f.name
        elif "normal" in path_str or "non" in path_str or "background" in path_str:
            dest = normal_target / f.name
        else:
            dest = fire_target / f.name
            
        if dest.exists():
            dest.unlink()
        shutil.move(str(f), str(dest))
        
    # Clean up train and validation folders
    for split_dir in ["train", "validation"]:
        p = firenet_dir / split_dir
        if p.exists():
            try:
                shutil.rmtree(p)
            except Exception as e:
                print(f"  Warning: could not delete {p}: {e}")
                
    print("[DONE] FireNet restructured.")


def main():
    restructure_flame()
    restructure_sard()
    restructure_firenet()

if __name__ == "__main__":
    main()
