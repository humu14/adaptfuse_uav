"""
Write the PANNs Cnn14_16k AudioSet tagger (Kong et al., 2020; mAP 0.438) into
weights/panns_cnn14_16k-0000N.safetensors as fp16 shards (< 90 MB each, so the repo
stays pushable to GitHub). The original 358 MB checkpoint comes from Zenodo record
3987831 and is downloaded once if no local copy is given.

    python scripts/export_audio.py [--src path/to/Cnn14_16k_mAP=0.438.pth]
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

DEMO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO))
import engine  # noqa: E402,F401
import torch  # noqa: E402

from engine.models import WEIGHTS_DIR, PANNsCnn14_16k, save_shards  # noqa: E402

URL = "https://zenodo.org/record/3987831/files/Cnn14_16k_mAP%3D0.438.pth?download=1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None)
    args = ap.parse_args()
    src = Path(args.src) if args.src else Path(torch.hub.get_dir()) / "Cnn14_16k_mAP=0.438.pth"
    if not src.exists():
        print(f"downloading {URL} -> {src}")
        src.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, src)
    sd = torch.load(src, map_location="cpu", weights_only=False)["model"]
    model = PANNsCnn14_16k()
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # the checkpoint also carries the training-time spec-augmenter buffers, which inference skips
    if missing:
        raise RuntimeError(f"missing keys: {missing}")
    print(f"unused checkpoint keys: {unexpected}")
    files = save_shards({k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()},
                        WEIGHTS_DIR, "panns_cnn14_16k")
    print("[ok] " + ", ".join(f"{f.name} ({f.stat().st_size / 1e6:.0f} MB)" for f in files))
    return 0


if __name__ == "__main__":
    sys.exit(main())
