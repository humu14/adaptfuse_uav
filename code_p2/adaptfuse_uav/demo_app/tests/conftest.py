import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Anaconda OpenMP clash
DEMO_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = Path(__file__).resolve().parents[4]          # Co-Sup/
sys.path.insert(0, str(DEMO_DIR))
