"""
Real failure-case gallery for thesis Chapter 5 (honest error analysis).
Runs the actual trained AdapFuse-v1 and baseline-audio checkpoints over the
real held-out test set and surfaces genuine misclassifications (no staged
examples). Intended to accompany the quantitative results with transparent
qualitative error analysis.

Usage (run from code_p2/adaptfuse_uav):
    python scripts/generate_qualitative_failures.py

Output -> e:\\Research Projects\\reserch writing\\Co-Sup\\images\\generated\\fig_13_failure_gallery.png
"""
import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets.multimodal_dataset import MultimodalDisasterDataset, bbox_collate_fn
from models.full_model import build_model

OUT = Path(r"e:\Research Projects\reserch writing\Co-Sup\images\generated")
OUT.mkdir(parents=True, exist_ok=True)

TEST_CSV = ROOT / "data/metadata/test.csv"
DISASTER_NAMES = ["Clean", "Fire/Smoke", "Collapse/Flood", "Other"]
RGB_MEAN = np.array([0.485, 0.456, 0.406])
RGB_STD = np.array([0.229, 0.224, 0.225])

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})


def denorm_rgb(t: torch.Tensor) -> np.ndarray:
    img = t.permute(1, 2, 0).numpy()
    img = img * RGB_STD + RGB_MEAN
    return np.clip(img, 0, 1)


def load_model(name, ckpt_name):
    model = build_model({"model": name})
    ckpt = torch.load(ROOT / f"outputs/checkpoints/{ckpt_name}", map_location="cpu")
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    result = model.load_state_dict(state, strict=True)
    print(f"[{name}] loaded cleanly: missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)}")
    model.eval()
    return model


@torch.no_grad()
def find_failures(model, ds, max_batches, has_rgb_required=True):
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0, collate_fn=bbox_collate_fn)
    failures = []
    for bi, batch in enumerate(dl):
        if bi >= max_batches:
            break
        d, v, rel = model(
            batch["rgb"], batch["thermal"], batch["audio"],
            batch["has_rgb"], batch["has_thermal"], batch["has_audio"],
        )
        probs = torch.softmax(d, dim=-1)
        pred = probs.argmax(dim=-1)
        conf = probs.max(dim=-1).values
        gt = batch["disaster_label"]
        for i in range(len(gt)):
            if pred[i].item() != gt[i].item():
                if has_rgb_required and batch["has_rgb"][i].item() == 0:
                    continue
                failures.append({
                    "rgb": batch["rgb"][i].clone(),
                    "true": int(gt[i]),
                    "pred": int(pred[i]),
                    "conf": float(conf[i]),
                    "sample_id": batch["sample_id"][i],
                    "source": batch["source"][i],
                })
    return failures


def main():
    ds_clean = MultimodalDisasterDataset(csv_path=str(TEST_CSV), split="test", corruption_prob=0.0)

    m_v1 = load_model("adaptfuse_v1", "adaptfuse_v1_best.pth")
    v1_failures = find_failures(m_v1, ds_clean, max_batches=40)  # ~1280 samples
    print(f"AdapFuse-v1 clean-test failures found: {len(v1_failures)}")

    m_audio = load_model("baseline_audio", "baseline_audio_best.pth")
    audio_failures = find_failures(m_audio, ds_clean, max_batches=15, has_rgb_required=True)
    print(f"Audio-only failures found: {len(audio_failures)}")

    rng = np.random.RandomState(3)
    picks = []
    if v1_failures:
        idxs = rng.choice(len(v1_failures), size=min(3, len(v1_failures)), replace=False)
        for i in idxs:
            f = v1_failures[i]
            f["tag"] = "AdapFuse-v1"
            picks.append(f)
    if audio_failures:
        idxs = rng.choice(len(audio_failures), size=min(3, len(audio_failures)), replace=False)
        for i in idxs:
            f = audio_failures[i]
            f["tag"] = "Audio-only (A3)"
            picks.append(f)

    n = len(picks)
    if n == 0:
        print("No failures found to display.")
        return

    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 4.0 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for i, (ax, f) in enumerate(zip(axes, picks)):
        ax.imshow(denorm_rgb(f["rgb"]))
        ax.axis("off")
        # model name once per row, not on every panel
        if i % ncols == 0:
            ax.text(-0.06, 0.5, f["tag"], transform=ax.transAxes, rotation=90,
                    ha="right", va="center", fontsize=10, fontweight="bold", color="#333333")
        ax.set_title(
            f"{DISASTER_NAMES[f['true']]} \u2192 {DISASTER_NAMES[f['pred']]} ({f['conf']:.2f})",
            fontsize=9, fontweight="bold", color="#B22222", pad=4,
        )
        ax.text(
            0.5, -0.02, f["sample_id"],
            transform=ax.transAxes, ha="center", va="top", fontsize=7, color="#888888",
        )

    for ax in axes[n:]:
        ax.axis("off")

    fig.tight_layout()

    out_path = OUT / "fig_13_failure_gallery.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved -> {out_path}")


if __name__ == "__main__":
    main()
