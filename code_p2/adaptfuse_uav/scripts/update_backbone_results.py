"""
Read backbone experiment test results and write BACKBONE_RESULTS.md.
Run after each training completes, or once at the end to get all results.
Usage:
    cd adaptfuse_uav
    python scripts/update_backbone_results.py
"""

import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
LOGS = ROOT / "outputs" / "logs"
OUT_MD = ROOT / "outputs" / "BACKBONE_RESULTS.md"

EXPERIMENTS = [
    # (exp_name, EXP_id, description, expected_gain)
    ("adaptfuse_v1",               "baseline",  "AdapFuse v1 (MobileNetV3 + ResNet18 + AudioCNN)",  "—"),
    ("adaptfuse_v1_panns",         "EXP-14",    "PANNS-CNN6 audio (AudioSet pretrained)",            "+5–10% audio F1"),
    ("adaptfuse_v1_efficientnet_rgb", "EXP-15", "EfficientNet-B0 RGB (1280d)",                      "+2–4% disaster F1"),
    ("adaptfuse_v1_efficientnet_thermal", "EXP-16", "EfficientNet-B0 thermal (1280d, 1-ch)",        "symmetric backbones"),
    ("adaptfuse_v1_mobilevit",     "EXP-17",    "MobileViT-XXS RGB (CNN+ViT, 320d)",               "aerial context"),
    ("adaptfuse_v1_gdblock",       "EXP-18",    "GDBlock audio (multi-scale dilated)",              "efficiency + citable"),
    ("adaptfuse_v1_domain_prompts","EXP-21",    "Domain prompt tokens (DPMamba-inspired)",          "+2–5% missing-modal"),
    ("adaptfuse_v1_spatial_attn",  "EXP-22",    "Spatial alignment attention before GAP",           "+1–2% under smoke"),
]


def load_result(exp_name: str) -> dict:
    path = LOGS / f"{exp_name}_test_results.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get("test", data)


def fmt(val, decimals=4):
    if val is None:
        return "—"
    return f"{val:.{decimals}f}"


def main():
    rows = []
    for exp_name, exp_id, desc, expected in EXPERIMENTS:
        res = load_result(exp_name)
        if res is None:
            d_f1 = v_f1 = comb = d_acc = "pending"
        else:
            d_f1  = fmt(res.get("disaster_f1"))
            v_f1  = fmt(res.get("victim_f1"))
            comb  = fmt(res.get("combined_f1"))
            d_acc = fmt(res.get("disaster_acc"))
        rows.append((exp_id, desc, d_f1, v_f1, comb, d_acc, expected))

    lines = [
        "# Backbone Experiment Results",
        "",
        f"_Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_",
        "",
        "## Test Set Results",
        "",
        "| EXP | Description | Disaster F1 | Victim F1 | Combined F1 | Disaster Acc | Expected gain |",
        "|-----|-------------|-------------|-----------|-------------|--------------|---------------|",
    ]
    for exp_id, desc, d_f1, v_f1, comb, d_acc, expected in rows:
        lines.append(f"| {exp_id} | {desc} | {d_f1} | {v_f1} | {comb} | {d_acc} | {expected} |")

    lines += [
        "",
        "## Status",
        "",
    ]
    for exp_id, desc, d_f1, *_ in rows:
        status = "✅ done" if d_f1 not in ("pending", "—") else "⏳ pending"
        lines.append(f"- **{exp_id}** {desc}: {status}")

    lines += [
        "",
        "## Run order",
        "",
        "```powershell",
        "cd adaptfuse_uav",
        "# EXP-14 (overnight first — safest, highest impact)",
        "python training/train.py --config configs/adaptfuse_v1_panns.yaml",
        "# EXP-18 (fast, ~1.5h)",
        "python training/train.py --config configs/adaptfuse_v1_gdblock.yaml",
        "# EXP-21 (fast, ~2h)",
        "python training/train.py --config configs/adaptfuse_v1_domain_prompts.yaml",
        "# EXP-22 (fast, ~2h)",
        "python training/train.py --config configs/adaptfuse_v1_spatial_attn.yaml",
        "# EXP-15 (batch=8, ~4-5h)",
        "python training/train.py --config configs/adaptfuse_v1_efficientnet_rgb.yaml",
        "# EXP-16 (batch=8, ~4-5h)",
        "python training/train.py --config configs/adaptfuse_v1_efficientnet_thermal.yaml",
        "# EXP-17 (batch=8, ~5-6h, requires: pip install timm)",
        "python training/train.py --config configs/adaptfuse_v1_mobilevit.yaml",
        "```",
        "",
    ]

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[SAVED] {OUT_MD}")

    # Also print summary to terminal
    print("\nBackbone Experiment Results:")
    print(f"{'EXP':<10} {'Combined F1':<14} {'Disaster F1':<14} {'Status'}")
    print("-" * 55)
    for exp_id, desc, d_f1, v_f1, comb, d_acc, _ in rows:
        status = "done" if comb not in ("pending", "—") else "pending"
        print(f"{exp_id:<10} {comb:<14} {d_f1:<14} {status}")


if __name__ == "__main__":
    main()
