"""
Real qualitative robustness figure for thesis Chapter 5 (Section 5.4).
Runs the actual trained AdapFuse-v1 checkpoint on ONE real test sample under
three real conditions (clean / smoke-occluded RGB / RGB sensor dropout),
using the production corruption pipeline (datasets/corruptions.py) and
dataset loader (datasets/multimodal_dataset.py). Reliability scores and
predictions shown are real forward-pass outputs, not illustrative numbers.

Usage (run from code_p2/adaptfuse_uav):
    python scripts/generate_qualitative_robustness.py

Output -> e:\\Research Projects\\reserch writing\\Co-Sup\\images\\generated\\fig_12_robustness_qualitative.png
"""
import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from datasets.multimodal_dataset import MultimodalDisasterDataset
from models.full_model import build_model

OUT = Path(r"e:\Research Projects\reserch writing\Co-Sup\images\generated")
OUT.mkdir(parents=True, exist_ok=True)

CKPT = ROOT / "outputs/checkpoints/adaptfuse_v1_best.pth"
TEST_CSV = ROOT / "data/metadata/test.csv"

DISASTER_NAMES = ["Clean", "Fire/Smoke", "Collapse/Flood", "Other Hazard"]
RGB_MEAN = np.array([0.485, 0.456, 0.406])
RGB_STD = np.array([0.229, 0.224, 0.225])

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})


def denorm_rgb(t: torch.Tensor) -> np.ndarray:
    img = t.permute(1, 2, 0).numpy()
    img = img * RGB_STD + RGB_MEAN
    return np.clip(img, 0, 1)


def denorm_thermal(t: torch.Tensor) -> np.ndarray:
    img = t.squeeze(0).numpy()
    lo, hi = np.percentile(img, 1), np.percentile(img, 99)
    img = np.clip((img - lo) / (hi - lo + 1e-6), 0, 1)
    return img


@torch.no_grad()
def run_forward(model, item):
    rgb = item["rgb"].unsqueeze(0)
    thermal = item["thermal"].unsqueeze(0)
    audio = item["audio"].unsqueeze(0)
    has_rgb = item["has_rgb"].unsqueeze(0)
    has_thermal = item["has_thermal"].unsqueeze(0)
    has_audio = item["has_audio"].unsqueeze(0)
    disaster_logits, victim_logits, reliability = model(
        rgb, thermal, audio, has_rgb, has_thermal, has_audio
    )
    probs = torch.softmax(disaster_logits, dim=-1)[0]
    pred = int(probs.argmax())
    conf = float(probs[pred])
    r = reliability[0].tolist()  # [r_rgb, r_thermal, r_audio]
    return pred, conf, r


def find_sample_idx(df):
    mask = (
        (df["source_dataset"] == "FLAME")
        & (df["disaster_label"] == 1)
        & (df["has_rgb"] == 1)
        & (df["has_thermal"] == 1)
        & (df["has_audio"] == 1)
    )
    idxs = df[mask].index.tolist()
    if not idxs:
        mask = (df["disaster_label"] == 1) & (df["has_rgb"] == 1) & (df["has_thermal"] == 1)
        idxs = df[mask].index.tolist()
    return idxs[0]


def main():
    import pandas as pd
    df = pd.read_csv(TEST_CSV)
    idx = find_sample_idx(df)
    print(f"Using test sample idx={idx}, sample_id={df.iloc[idx]['sample_id']}")

    model = build_model({"model": "adaptfuse_v1"})
    ckpt = torch.load(CKPT, map_location="cpu")
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    result = model.load_state_dict(state, strict=True)
    print(f"Loaded checkpoint cleanly: missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)}")
    model.eval()

    ds_clean = MultimodalDisasterDataset(csv_path=str(TEST_CSV), split="test", corruption_prob=0.0)
    ds_smoke = MultimodalDisasterDataset(
        csv_path=str(TEST_CSV), split="test",
        corruption_prob=1.0, corruption_type="smoke_overlay", corruption_severity=5,
    )

    item_clean = ds_clean[idx]
    item_smoke = ds_smoke[idx]
    item_dropout = {k: v.clone() if torch.is_tensor(v) else v for k, v in item_clean.items()}
    item_dropout["rgb"] = torch.zeros_like(item_dropout["rgb"])
    item_dropout["has_rgb"] = torch.tensor(0.0)

    conditions = [
        ("Clean", item_clean, denorm_rgb(item_clean["rgb"]), True),
        ("Smoke (severity 5)", item_smoke, denorm_rgb(item_smoke["rgb"]), True),
        ("RGB dropout", item_dropout, None, False),
    ]

    fig = plt.figure(figsize=(12.5, 5.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[2.4, 1.0], hspace=0.35, wspace=0.28)

    thermal_img = denorm_thermal(item_clean["thermal"])

    for col, (title, item, rgb_img, show_rgb) in enumerate(conditions):
        pred, conf, r = run_forward(model, item)

        ax_img = fig.add_subplot(gs[0, col])
        if show_rgb:
            ax_img.imshow(rgb_img)
        else:
            ax_img.imshow(np.full((224, 224, 3), 0.15))
            ax_img.text(112, 112, "RGB off", ha="center", va="center", color="#FF6B6B", fontsize=11, fontweight="bold")
        # thermal inset (always available/used by the model in this scenario)
        inset = ax_img.inset_axes([0.66, 0.02, 0.32, 0.32])
        inset.imshow(thermal_img, cmap="inferno")
        inset.set_xticks([]); inset.set_yticks([])
        for s in inset.spines.values():
            s.set_edgecolor("white"); s.set_linewidth(1.2)
        ax_img.set_xticks([]); ax_img.set_yticks([])
        ax_img.set_title(title, fontsize=10.5, fontweight="bold")

        pred_color = "#2ECC71" if pred == 1 else "#D62728"
        ax_img.text(
            0.5, -0.06, f"{DISASTER_NAMES[pred]}  ({conf:.2f})",
            transform=ax_img.transAxes, ha="center", va="top", fontsize=9,
            color=pred_color, fontweight="bold",
        )

        ax_bar = fig.add_subplot(gs[1, col])
        labels = ["RGB", "Pseudo-T", "Audio"]
        colors = ["#2271B5", "#C0392B", "#0D9E73"]
        y = np.arange(3)
        ax_bar.barh(y, r, color=colors, height=0.55)
        for yi, val in zip(y, r):
            ax_bar.text(min(val + 0.03, 0.85), yi, f"{val:.3f}", va="center", fontsize=8)
        ax_bar.set_yticks(y); ax_bar.set_yticklabels(labels if col == 0 else [], fontsize=8.5)
        ax_bar.set_xlim(0, 1.05)
        ax_bar.invert_yaxis()
        if col == 1:
            ax_bar.set_xlabel("RUE gate $r_i$", fontsize=8)
        ax_bar.spines["top"].set_visible(False)
        ax_bar.spines["right"].set_visible(False)

    handles = [mpatches.Patch(color="white", ec="white", label="Inset (bottom-right): thermal stream")]
    fig.tight_layout()

    out_path = OUT / "fig_12_robustness_qualitative.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved -> {out_path}")


if __name__ == "__main__":
    main()
