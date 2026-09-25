"""
Real qualitative D-Fire detection figure for the thesis Results chapter.
Runs the actual trained YOLO26s(CLAHE) checkpoint (77.10% mAP50, top row of
Table 5.1) on real D-Fire test images and draws real predicted boxes against
real ground-truth boxes. No synthetic/staged boxes.

Output -> e:\\Research Projects\\reserch writing\\Co-Sup\\images\\generated\\fig_11_dfire_qualitative.png
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).parent.parent
CKPT = ROOT / "outputs/accuracy_study/accuracy_yolo26s_p2_clahe_640/weights/best.pt"
IMG_DIR = ROOT / "data/datasets/dfire/images/test"
LBL_DIR = ROOT / "data/datasets/dfire/labels/test"
OUT = Path(r"e:\Research Projects\reserch writing\Co-Sup\images\generated")
OUT.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {0: "smoke", 1: "fire"}
CLASS_COLOR = {0: "#3C5488", 1: "#D62728"}  # smoke=blue, fire=red

SAMPLES = [
    ("WEB10547", "Smoke + fire (night)"),
    ("AoF08364", "Smoke plume"),
    ("WEB10719", "Flame (close)"),
    ("AoF08181", "Distant smoke"),
    ("WEB10607", "Smoke + fire (day)"),
    ("WEB10912", "Flame (wide)"),
]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
})


def load_gt_boxes(stem, w, h):
    p = LBL_DIR / f"{stem}.txt"
    boxes = []
    if not p.exists():
        return boxes
    for line in p.read_text().strip().splitlines():
        c, cx, cy, bw, bh = line.split()
        c = int(c); cx, cy, bw, bh = map(float, (cx, cy, bw, bh))
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        boxes.append((c, x1, y1, x2, y2))
    return boxes


def main():
    model = YOLO(str(CKPT))

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2))
    axes = axes.flatten()

    for ax, (stem, label) in zip(axes, SAMPLES):
        img_path = IMG_DIR / f"{stem}.jpg"
        im = Image.open(img_path).convert("RGB")
        w, h = im.size

        result = model.predict(str(img_path), conf=0.25, iou=0.5, verbose=False)[0]

        ax.imshow(im)
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.axis("off")

        # Ground truth: dashed outline
        for c, x1, y1, x2, y2 in load_gt_boxes(stem, w, h):
            rect = mpatches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=1.6, edgecolor="#2ECC71", facecolor="none",
                linestyle=(0, (4, 2)), zorder=3,
            )
            ax.add_patch(rect)

        # Predictions: solid outline + confidence label
        boxes = result.boxes
        placed = []
        if boxes is not None:
            for b in sorted(boxes, key=lambda b: -float(b.conf.item())):
                cls = int(b.cls.item())
                conf = float(b.conf.item())
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                color = CLASS_COLOR.get(cls, "#888888")
                rect = mpatches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2.2, edgecolor=color, facecolor="none", zorder=4,
                )
                ax.add_patch(rect)
                # move a tag below its box if it would sit on an earlier tag
                ty = max(y1 - 4, 8)
                if any(abs(x1 - px) < 0.12 * w and abs(ty - py) < 0.06 * h for px, py in placed):
                    ty = min(y2 + 0.05 * h, h - 4)
                placed.append((x1, ty))
                ax.text(
                    x1, ty, f"{CLASS_NAMES.get(cls,'?')} {conf:.2f}",
                    color="white", fontsize=7.5, fontweight="bold", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.15", fc=color, ec="none", alpha=0.92),
                )

        ax.set_title(label, fontsize=9.5, fontweight="bold", color="#222222", pad=4)
        for spine in ax.spines.values():
            spine.set_visible(True)

    # Legend
    handles = [
        mpatches.Patch(edgecolor="#2ECC71", facecolor="none", linestyle=(0, (4, 2)), linewidth=1.6, label="Ground truth"),
        mpatches.Patch(edgecolor=CLASS_COLOR[1], facecolor="none", linewidth=2.2, label="Fire"),
        mpatches.Patch(edgecolor=CLASS_COLOR[0], facecolor="none", linewidth=2.2, label="Smoke"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, -0.01))

    fig.tight_layout(rect=[0, 0.035, 1, 1])

    out_path = OUT / "fig_11_dfire_qualitative.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved -> {out_path}")


if __name__ == "__main__":
    main()
