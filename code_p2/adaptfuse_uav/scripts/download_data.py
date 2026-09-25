"""
AdapFuse-UAV: Dataset download script.
Downloads real datasets: FLAME, AIDER, SARD, FireNet, and ESC-50.
Uses Kaggle API for FLAME, AIDER, SARD, and FireNet.
"""

import os
import sys
import shutil
import zipfile
import tarfile
import requests
import gc
from pathlib import Path
from tqdm import tqdm


RAW_DIR = Path(__file__).parent.parent / "datasets" / "raw"


def download_file(url: str, dest: Path, desc: str = "Downloading"):
    """Download a file with tqdm progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[SKIP] {dest.name} already exists.")
        return

    response = requests.get(url, stream=True, timeout=120)
    total = int(response.headers.get("content-length", 0))

    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=desc
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))


def extract_zip(src: Path, dest_dir: Path):
    print(f"[EXTRACT] {src.name} -> {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src, "r") as z:
        members = z.namelist()
        for member in tqdm(members, desc=f"Extracting {src.name}"):
            z.extract(member, dest_dir)


def download_esc50():
    """Download ESC-50 dataset from GitHub."""
    print("\n=== Downloading ESC-50 ===")
    url = "https://github.com/karoldvl/ESC-50/archive/master.zip"
    dest = RAW_DIR / "ESC50" / "ESC-50-master.zip"
    download_file(url, dest, desc="ESC-50")
    if dest.exists() and not (RAW_DIR / "ESC50" / "ESC-50-master").exists():
        extract_zip(dest, RAW_DIR / "ESC50")
    print("[DONE] ESC-50 downloaded.")


def is_dataset_populated(dest_dir: Path, expected_subdirs: list) -> bool:
    """Check if the dataset directory is already populated with files."""
    if not dest_dir.exists():
        return False
    for subdir in expected_subdirs:
        sub_path = dest_dir / subdir
        if not sub_path.exists():
            return False
        # Check if there are any files inside
        files = list(sub_path.rglob("*"))
        # Filter files only
        files = [f for f in files if f.is_file()]
        if len(files) == 0:
            return False
    return True


def download_kaggle_dataset(dataset_ref: str, dest_dir: Path, desc: str, expected_subdirs: list):
    """Download a dataset from Kaggle using the Python API."""
    print(f"\n=== Downloading {desc} from Kaggle ({dataset_ref}) ===")
    
    if is_dataset_populated(dest_dir, expected_subdirs):
        print(f"[SKIP] {desc} already downloaded and structured.")
        return False

    dest_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        import kaggle
        api = kaggle.KaggleApi()
        api.authenticate()
        
        zip_name = dataset_ref.split('/')[-1] + ".zip"
        zip_path = dest_dir / zip_name
        
        if zip_path.exists():
            if not zipfile.is_zipfile(zip_path):
                print(f"[WARNING] Zip file {zip_name} is corrupted or invalid. Deleting and re-downloading...")
                try:
                    zip_path.unlink()
                except Exception as e:
                    print(f"Warning: could not delete invalid zip file {zip_path}: {e}")
        
        if zip_path.exists():
            print(f"[SKIP] Zip file {zip_name} already exists.")
        else:
            print(f"Downloading {dataset_ref} to {dest_dir}...")
            api.dataset_download_files(dataset_ref, path=str(dest_dir), unzip=False, quiet=False)
            
        gc.collect()
            
        if zip_path.exists():
            print(f"Extracting {zip_name} to {dest_dir}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                members = zip_ref.namelist()
                for member in tqdm(members, desc=f"Extracting {desc}"):
                    zip_ref.extract(member, dest_dir)
            
            try:
                gc.collect()
                os.remove(zip_path)
            except Exception as e:
                print(f"Warning: could not delete zip file {zip_path}: {e}")
        print(f"[DONE] {desc} downloaded and extracted.")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to download {desc}: {e}")
        print(f"[WARN] Skipping {desc} and continuing with the remaining datasets.")
        return False


def restructure_aider():
    aider_dir = RAW_DIR / "AIDER"
    aider_full = aider_dir / "AIDER_full"
    if aider_full.exists():
        print("[RESTRUCTURE] Moving AIDER files from AIDER_full...")
        for item in aider_full.iterdir():
            dest = aider_dir / item.name
            try:
                if dest.exists():
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()
                shutil.move(str(item), str(dest))
            except Exception as e:
                print(f"Warning: could not move AIDER item {item} to {dest}: {e}")
        try:
            shutil.rmtree(aider_full)
        except Exception as e:
            print(f"Warning: could not delete AIDER_full folder: {e}")
        print("[RESTRUCTURE] AIDER restructured successfully.")


def restructure_sard():
    sard_dir = RAW_DIR / "SARD"
    sard_images = sard_dir / "images"
    sard_images.mkdir(parents=True, exist_ok=True)
    
    print("[RESTRUCTURE] Moving SARD images into SARD/images...")
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    for file_path in list(sard_dir.rglob("*")):
        try:
            if file_path.is_file() and file_path.suffix.lower() in img_exts:
                if not file_path.is_relative_to(sard_images):
                    dest_path = sard_images / file_path.name
                    if dest_path.exists():
                        dest_path.unlink()
                    shutil.move(str(file_path), str(dest_path))
        except Exception as e:
            print(f"Warning: error moving SARD file {file_path}: {e}")
                
    # Clean up other subdirectories except images
    for item in sard_dir.iterdir():
        try:
            if item.is_dir() and item.name != "images":
                shutil.rmtree(item)
            elif item.is_file() and item.name != "images" and item.suffix.lower() != ".zip":
                item.unlink()
        except Exception as e:
            print(f"Warning: could not delete item {item} in SARD: {e}")
    print("[RESTRUCTURE] SARD restructured successfully.")


def restructure_flame():
    flame_dir = RAW_DIR / "FLAME"
    print("[RESTRUCTURE] Organizing FLAME folders...")
    
    temp_dir = RAW_DIR / "FLAME_temp"
    if temp_dir.exists():
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Warning: could not clean temp dir {temp_dir}: {e}")
            
    try:
        shutil.move(str(flame_dir), str(temp_dir))
    except Exception as e:
        print(f"Warning: could not move FLAME to temp: {e}")
        return
    
    # Create target directories
    for split in ["Training", "Test"]:
        for cls in ["fire", "non-fire"]:
            (flame_dir / split / cls).mkdir(parents=True, exist_ok=True)
            
    # Move files
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    for file_path in temp_dir.rglob("*"):
        try:
            if file_path.is_file() and file_path.suffix.lower() in img_exts:
                path_str = str(file_path).lower()
                
                if "training" in path_str:
                    split = "Training"
                elif "test" in path_str:
                    split = "Test"
                else:
                    continue
                    
                if "non-fire" in path_str or "non_fire" in path_str or "nonfire" in path_str:
                    cls = "non-fire"
                elif "fire" in path_str:
                    cls = "fire"
                else:
                    continue
                    
                dest = flame_dir / split / cls / file_path.name
                shutil.move(str(file_path), str(dest))
        except Exception as e:
            print(f"Warning: error moving FLAME file {file_path}: {e}")
            
    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"Warning: could not delete temp FLAME dir: {e}")
    print("[RESTRUCTURE] FLAME restructured successfully.")


def restructure_c2a():
    """Restructure C2A dataset into disaster-labeled subdirectories."""
    c2a_dir = RAW_DIR / "C2A"
    if not c2a_dir.exists():
        print("[SKIP] C2A directory not found, nothing to restructure.")
        return

    print("[RESTRUCTURE] Organizing C2A into disaster-class subfolders...")
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    # C2A is organized as: C2A/<disaster_class>/<images>
    # Disaster class folder names may vary; map them to our label schema
    C2A_CLASS_MAP = {
        "fire": "fire",
        "fire_smoke": "fire",
        "smoke": "fire",
        "flood": "flood",
        "flooded": "flood",
        "collapse": "collapsed",
        "collapsed": "collapsed",
        "rubble": "collapsed",
        "building": "collapsed",
        "traffic": "traffic",
        "accident": "traffic",
    }

    # Create target class directories
    for cls in ["fire", "flood", "collapsed", "traffic"]:
        (c2a_dir / cls).mkdir(parents=True, exist_ok=True)

    moved = 0
    for item in list(c2a_dir.iterdir()):
        if not item.is_dir():
            continue
        # Determine target class from folder name
        folder_lower = item.name.lower()
        target_cls = None
        for keyword, cls in C2A_CLASS_MAP.items():
            if keyword in folder_lower:
                target_cls = cls
                break
        if target_cls is None:
            # If already one of our target dirs, skip
            if folder_lower in {"fire", "flood", "collapsed", "traffic"}:
                continue
            print(f"  [SKIP] Unknown C2A subfolder: {item.name}")
            continue

        target_dir = c2a_dir / target_cls
        for img_file in item.rglob("*"):
            if img_file.is_file() and img_file.suffix.lower() in img_exts:
                dest = target_dir / img_file.name
                # Avoid name collisions
                if dest.exists():
                    stem, suffix = img_file.stem, img_file.suffix
                    dest = target_dir / f"{stem}_{moved}{suffix}"
                try:
                    shutil.move(str(img_file), str(dest))
                    moved += 1
                except Exception as e:
                    print(f"  Warning: could not move {img_file}: {e}")
        # Remove now-empty source directory
        try:
            if item.name.lower() not in {"fire", "flood", "collapsed", "traffic"}:
                shutil.rmtree(item)
        except Exception as e:
            print(f"  Warning: could not remove folder {item}: {e}")

    print(f"[RESTRUCTURE] C2A restructured — {moved} images moved.")


def restructure_firenet():
    firenet_dir = RAW_DIR / "FireNet"
    print("[RESTRUCTURE] Organizing FireNet folders...")
    
    temp_dir = RAW_DIR / "FireNet_temp"
    if temp_dir.exists():
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Warning: could not clean temp FireNet: {e}")
            
    try:
        shutil.move(str(firenet_dir), str(temp_dir))
    except Exception as e:
        print(f"Warning: could not move FireNet to temp: {e}")
        return
    
    # Create target directories
    fire_target = firenet_dir / "fire"
    normal_target = firenet_dir / "normal"
    smoke_target = firenet_dir / "smoke"
    fire_target.mkdir(parents=True, exist_ok=True)
    normal_target.mkdir(parents=True, exist_ok=True)
    smoke_target.mkdir(parents=True, exist_ok=True)
    
    # Move files
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    for file_path in temp_dir.rglob("*"):
        try:
            if file_path.is_file() and file_path.suffix.lower() in img_exts:
                path_str = file_path.name.lower()
                if "smoke" in path_str:
                    dest = smoke_target / file_path.name
                elif "normal" in path_str or "non" in path_str or "background" in path_str:
                    dest = normal_target / file_path.name
                else:
                    dest = fire_target / file_path.name
                    
                shutil.move(str(file_path), str(dest))
        except Exception as e:
            print(f"Warning: error moving FireNet file {file_path}: {e}")
            
    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"Warning: could not delete temp FireNet dir: {e}")
    print("[RESTRUCTURE] FireNet restructured successfully.")


def download_flame3():
    """FLAME 3: real UAV wildfire RGB + radiometric thermal (Dec 2024)."""
    print("\n=== Checking FLAME 3 Dataset ===")
    dest_dir = RAW_DIR / "FLAME3"
    if is_dataset_populated(dest_dir, ["fire", "no_fire"]):
        print("[SKIP] FLAME3 already downloaded and populated.")
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        download_kaggle_dataset(
            "mohammadreza13/flame-3-wildfire-uav-dataset",
            dest_dir,
            "FLAME 3 (Wildfire UAV Radiometric)",
            ["fire", "no_fire"]
        )
    except Exception as e:
        print(f"[INFO] FLAME 3 Kaggle download note: {e}")
        print("  Manual source: https://ieee-dataport.org/open-access/flame-3-radiometric-thermal-uav-imagery-wildfire-management")


def download_droneaudioset():
    """DroneAudioset (NeurIPS 2025, MIT): 23.5h real UAV SAR audio."""
    print("\n=== Checking DroneAudioset (Real UAV SAR Audio) ===")
    dest_dir = RAW_DIR / "DroneAudioset"
    if dest_dir.exists() and any(dest_dir.rglob("*.wav")):
        print("[SKIP] DroneAudioset already populated.")
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
        print("[DOWNLOAD] Downloading DroneAudioset from HuggingFace...")
        snapshot_download(
            repo_id="ahlab-drone-project/DroneAudioSet",
            repo_type="dataset",
            local_dir=str(dest_dir),
            ignore_patterns=["*.parquet", "*.arrow", "*.json.gz"],
        )
        print("[DONE] DroneAudioset downloaded.")
    except Exception as e:
        print(f"[INFO] HuggingFace download exception for DroneAudioset: {e}")
        print("  Can also be fetched via: git clone https://huggingface.co/datasets/ahlab-drone-project/DroneAudioSet")


def setup_dregon_manual():
    """DREGON: print instructions. Registration required at dregon.inria.fr."""
    dregon_dir = RAW_DIR / "DREGON"
    video_exts = {".mp4", ".avi", ".mov", ".mkv"}
    ir_dir = dregon_dir / "infrared"
    has_data = dregon_dir.exists() and ir_dir.exists() and any(
        f.suffix.lower() in video_exts for f in ir_dir.rglob("*")
    )
    if has_data:
        print(f"[DREGON] Real tri-modal synchronized data found at {dregon_dir}.")
        return
    print("[DREGON] Notice: Manual registration required at https://dregon.inria.fr/")
    print(f"  Extract to: {dregon_dir}/infrared/  visible/  audio/")


if __name__ == "__main__":
    print("=" * 60)
    print("AdapFuse-UAV Real Dataset Downloader & Restructurer")
    print("=" * 60)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Download ESC-50 (Audio)
    download_esc50()

    # 2. Download AIDER (RGB Disaster Imagery)
    aider_subdirs = ["fire", "flooded_areas", "collapsed_building", "traffic_incident", "normal"]
    if download_kaggle_dataset("samik2005/aider-dataset", RAW_DIR / "AIDER", "AIDER", aider_subdirs):
        restructure_aider()

    # 3. Download SARD (RGB Search and Rescue Imagery)
    sard_subdirs = ["images"]
    if download_kaggle_dataset("nikolasgegenava/sard-search-and-rescue", RAW_DIR / "SARD", "SARD", sard_subdirs):
        restructure_sard()
    else:
        restructure_sard()

    # 4. Download FLAME (RGB Fire Classification Imagery)
    flame_subdirs = ["Training/fire", "Training/non-fire", "Test/fire", "Test/non-fire"]
    if download_kaggle_dataset("smrutisanchitadas/flame-dataset-fire-classification", RAW_DIR / "FLAME", "FLAME", flame_subdirs):
        restructure_flame()

    # 5. Download FireNet (RGB Fire and Smoke Imagery)
    firenet_subdirs = ["fire", "normal", "smoke"]
    if download_kaggle_dataset("dataclusterlabs/fire-and-smoke-dataset", RAW_DIR / "FireNet", "FireNet", firenet_subdirs):
        restructure_firenet()

    # 6. Download C2A
    c2a_subdirs = ["fire", "flood", "collapsed", "traffic"]
    if download_kaggle_dataset(
        "ragibaminnihal/c2a-dataset-human-detection-in-disaster-scenarios",
        RAW_DIR / "C2A",
        "C2A (Human Detection in Disasters)",
        c2a_subdirs
    ):
        restructure_c2a()
    else:
        restructure_c2a()

    # 7. FLAME 3 (Radiometric Thermal + RGB)
    download_flame3()

    # 8. DroneAudioset (Real UAV SAR Audio)
    download_droneaudioset()

    # 9. DREGON Tri-Modal
    setup_dregon_manual()

    print("\n" + "=" * 60)
    print("Real dataset setup complete!")
    print(f"Raw data directory: {RAW_DIR.resolve()}")
    print("\nNext step: Run scripts/build_metadata.py to create CSVs")
    print("=" * 60)
