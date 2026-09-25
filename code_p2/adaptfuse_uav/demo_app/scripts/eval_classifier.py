"""
Measure the demo's scene classifier on the grouped split exactly as the app feeds it
(one visual modality + audio when the row has it), stage by stage, and tune the
per-condition class bias on the val split.

Run from code_p2/adaptfuse_uav/demo_app:
    python scripts/eval_classifier.py                 # full run (~10-20 min on GTX 960)
    python scripts/eval_classifier.py --limit 300     # quick smoke run, writes nothing to weights/
    python scripts/eval_classifier.py --compare-fp32  # also check fp16 vs training checkpoints

Outputs: eval/classifier_eval.{md,json}; weights/calibration.json (full runs only).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

DEMO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO))
import engine  # noqa: E402,F401
import cv2  # noqa: E402
import librosa  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import f1_score, precision_recall_fscore_support  # noqa: E402

from engine.classifier import apply_bias, member_outputs  # noqa: E402
from engine.labels import DISASTER_CLASSES  # noqa: E402
from engine.models import CLASSIFIER_FILES, PROJECT_DIR, WEIGHTS_DIR, load_member  # noqa: E402
from engine.pipeline import _logmel, _rgb_tensor, _thermal_tensor  # noqa: E402

SPLIT_DIR = PROJECT_DIR / "data" / "metadata_grouped" / "seed_42"
TRAIN_CKPTS = {
    "adaptfuse_v1_grouped_s41.pth": PROJECT_DIR / "outputs/grouped_repeated/seed_41/checkpoints/adaptfuse_v1_grouped_best.pth",
    "adaptfuse_v1_grouped_s42.pth": PROJECT_DIR / "outputs/grouped_repeated/seed_42/checkpoints/adaptfuse_v1_grouped_best.pth",
}
OUT = DEMO / "eval"
CONDITIONS = ("rgb", "thermal")
CHAIN = ("seed42", "ensemble", "ensemble+tta", "ensemble+tta+bias")   # adoption order
BATCH = 32


def load_rows(split: str, condition: str, limit: int | None) -> pd.DataFrame:
    df = pd.read_csv(SPLIT_DIR / f"{split}.csv")
    df = df[df["has_rgb" if condition == "rgb" else "has_thermal"] == 1].reset_index(drop=True)
    if limit:
        # stratified head so every class is present in smoke runs
        df = df.groupby("disaster_label", group_keys=False).head(max(1, limit // 4)).reset_index(drop=True)
    return df


_AUDIO_CACHE: dict = {}


def audio_tensor(row) -> tuple:
    if not int(row["has_audio"]) or not isinstance(row["audio_path"], str):
        return torch.zeros(1, 1, 64, 63), 0.0
    p = row["audio_path"]
    if p not in _AUDIO_CACHE:
        y, _ = librosa.load(p, sr=16000, duration=2.0, mono=True)   # training: first 2 s, end-padded
        y = np.pad(y, (0, max(0, 32000 - len(y))))[:32000]
        _AUDIO_CACHE[p] = _logmel(y)
    return _AUDIO_CACHE[p], 1.0


def visual_tensors(row, condition: str):
    if condition == "rgb":
        return _rgb_tensor(cv2.imread(row["rgb_path"], cv2.IMREAD_COLOR)), torch.zeros(1, 1, 224, 224), 1.0, 0.0
    img = cv2.imread(row["thermal_path"], cv2.IMREAD_COLOR)
    return torch.zeros(1, 3, 224, 224), _thermal_tensor(img), 0.0, 1.0


def member_probs(members: dict, df: pd.DataFrame, condition: str, device) -> dict:
    """{member_name: {"orig": {head: (N,K)}, "flip": {...}}} as numpy."""
    acc = {n: {"orig": {}, "flip": {}} for n in members}
    for s in range(0, len(df), BATCH):
        rows = [df.iloc[i] for i in range(s, min(s + BATCH, len(df)))]
        vis = [visual_tensors(r, condition) for r in rows]
        aud = [audio_tensor(r) for r in rows]
        rgb = torch.cat([v[0] for v in vis]); th = torch.cat([v[1] for v in vis])
        au = torch.cat([a[0] for a in aud])
        has = torch.tensor([[v[2], v[3], a[1]] for v, a in zip(vis, aud)], dtype=torch.float32)
        for view, (r_, t_) in (("orig", (rgb, th)), ("flip", (torch.flip(rgb, [3]), torch.flip(th, [3])))):
            for name, m in members.items():
                o = member_outputs(m, r_.to(device), t_.to(device), au.to(device), has.to(device))
                for k in ("disaster", "victim"):
                    acc[name][view].setdefault(k, []).append(o[k].numpy())
        print(f"\r  {condition}: {min(s + BATCH, len(df))}/{len(df)}", end="", flush=True)
    print()
    return {n: {v: {k: np.concatenate(x) for k, x in d.items()} for v, d in views.items()}
            for n, views in acc.items()}


def stage_probs(P: dict, stage: str, bias=None) -> dict:
    s41, s42 = "adaptfuse_v1_grouped_s41.pth", "adaptfuse_v1_grouped_s42.pth"
    if stage == "seed42":
        return P[s42]["orig"]
    if stage == "seed41":
        return P[s41]["orig"]
    views = [P[s41]["orig"], P[s42]["orig"]]
    if stage.startswith("ensemble+tta"):
        views += [P[s41]["flip"], P[s42]["flip"]]
    out = {k: np.nan_to_num(np.nanmean([v[k] for v in views], axis=0), nan=0.25 if k == "disaster" else 0.5)
           for k in ("disaster", "victim")}
    if stage == "ensemble+tta+bias":
        out["disaster"] = apply_bias(out["disaster"], bias)
    return out


def metrics(p: dict, df: pd.DataFrame) -> dict:
    yd, yv = df["disaster_label"].to_numpy(), df["victim_label"].to_numpy()
    pd_, pv = p["disaster"].argmax(1), p["victim"].argmax(1)
    pr, rc, f1, sup = precision_recall_fscore_support(yd, pd_, labels=[0, 1, 2, 3], zero_division=0)
    return {
        "disaster_macro_f1": float(f1_score(yd, pd_, labels=[0, 1, 2, 3], average="macro", zero_division=0)),
        "victim_macro_f1": float(f1_score(yv, pv, labels=[0, 1], average="macro", zero_division=0)),
        "per_class": {DISASTER_CLASSES[i]: {"precision": float(pr[i]), "recall": float(rc[i]),
                                            "f1": float(f1[i]), "support": int(sup[i])} for i in range(4)},
        "n": int(len(df)),
    }


PRECISION_FLOOR = 0.5   # a bias may not buy macro-F1 by flooding a class with false alarms


def tune_bias(p: np.ndarray, y: np.ndarray) -> list:
    """Coordinate search on classes 1..3 (class 0 fixed at 0: softmax is shift-invariant).

    A candidate is rejected if any class it predicts has val precision below
    PRECISION_FLOOR. A class already below the floor without bias may stay there, but
    only if the bias neither adds predictions of it nor lowers its precision.
    """
    b = np.zeros(4)
    grid = np.round(np.arange(-2.0, 2.0001, 0.1), 2)
    base_pred = p.argmax(1)
    base_n = [int((base_pred == c).sum()) for c in range(4)]
    base_prec = [((base_pred == c) & (y == c)).sum() / base_n[c] if base_n[c] else 1.0 for c in range(4)]

    def score(bb):
        pred = apply_bias(p, bb).argmax(1)
        for c in range(4):
            n = int((pred == c).sum())
            if not n:
                continue
            prec = ((pred == c) & (y == c)).sum() / n
            if prec < PRECISION_FLOOR and (base_prec[c] >= PRECISION_FLOOR or n > base_n[c] or prec < base_prec[c]):
                return -1.0
        return f1_score(y, pred, labels=[0, 1, 2, 3], average="macro", zero_division=0)

    best = score(b)
    for _ in range(3):
        for c in (1, 2, 3):
            for g in grid:
                trial = b.copy(); trial[c] = g
                s = score(trial)
                if s > best + 1e-9:
                    best, b = s, trial
    return [float(x) for x in b]


def compare_fp32(members: dict, device, limit: int | None) -> dict:
    df = load_rows("test", "rgb", limit)
    fp32 = {n: load_member(TRAIN_CKPTS[n], device) for n in members}
    a = member_probs(members, df, "rgb", device)
    b = member_probs(fp32, df, "rgb", device)
    res = {}
    for n in members:
        pa, pb = a[n]["orig"]["disaster"], b[n]["orig"]["disaster"]
        ok = ~(np.isnan(pa).any(1) | np.isnan(pb).any(1))
        res[n] = {"argmax_agreement": float((pa[ok].argmax(1) == pb[ok].argmax(1)).mean()),
                  "max_abs_prob_diff": float(np.abs(pa[ok] - pb[ok]).max()), "n": int(ok.sum()),
                  "nan_rows_fp16": int(np.isnan(pa).any(1).sum()), "nan_rows_fp32": int(np.isnan(pb).any(1).sum())}
    return res


def write_report(results: dict, chosen: str, cal: dict, fp: dict | None, secs: float, limit) -> None:
    OUT.mkdir(exist_ok=True)
    (OUT / "classifier_eval.json").write_text(json.dumps(
        {"results": results, "chosen_stage": chosen, "calibration": cal, "fp16_check": fp,
         "limit": limit, "seconds": round(secs, 1)}, indent=1))
    L = ["# Scene classifier evaluation (grouped split, seed_42)", "",
         "Input condition as in the demo app: **one** visual modality + audio where the row has it. "
         "Bias tuned on **val**; numbers below are **test**. "
         "The adoption rule compares stages on test macro-F1, so the chosen stage is selected on test.", ""]
    if limit:
        L += [f"> Smoke run with --limit {limit}; not a real measurement.", ""]
    for c in CONDITIONS:
        L += [f"## Condition: {c}  (n = {results[c]['seed42']['n']})", "",
              "| Stage | Disaster macro-F1 | F1 normal | F1 fire/smoke | F1 collapse/flood | F1 other | Victim macro-F1 |",
              "|---|---|---|---|---|---|---|"]
        for st in ("seed41",) + CHAIN:
            m = results[c][st]; pc = m["per_class"]
            mark = " **(in app)**" if st == chosen else ""
            L.append(f"| {st}{mark} | {m['disaster_macro_f1']:.4f} | " + " | ".join(
                f"{pc[k]['f1']:.4f}" for k in DISASTER_CLASSES) + f" | {m['victim_macro_f1']:.4f} |")
        L += ["", f"Per-class precision / recall for the app stage (`{chosen}`):", "",
              "| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
        for k, v in results[c][chosen]["per_class"].items():
            L.append(f"| {k} | {v['precision']:.4f} | {v['recall']:.4f} | {v['f1']:.4f} | {v['support']} |")
        L.append("")
    L += ["## Calibration written to `weights/calibration.json`", "", "```json", json.dumps(cal, indent=1), "```", ""]
    if fp:
        L += ["## fp16 vs fp32 check (test, rgb)", "",
              "| Member | Argmax agreement | Max abs Δp | n (finite) | NaN rows fp16 / fp32 |", "|---|---|---|---|---|"]
        L += [f"| {n} | {v['argmax_agreement']:.5f} | {v['max_abs_prob_diff']:.5f} | {v['n']} | "
              f"{v['nan_rows_fp16']} / {v['nan_rows_fp32']} |" for n, v in fp.items()]
        L.append("")
    L.append(f"Run time: {secs / 60:.1f} min.")
    (OUT / "classifier_eval.md").write_text("\n".join(L), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--compare-fp32", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    dev = torch.device(args.device)
    members = {n: load_member(WEIGHTS_DIR / n, dev) for n in CLASSIFIER_FILES}

    results, bias = {}, {}
    for c in CONDITIONS:
        print(f"[val] {c}")
        dv = load_rows("val", c, args.limit)
        Pv = member_probs(members, dv, c, dev)
        bias[c] = tune_bias(stage_probs(Pv, "ensemble+tta")["disaster"], dv["disaster_label"].to_numpy())
        print(f"  bias {c}: {bias[c]}")
        print(f"[test] {c}")
        dt = load_rows("test", c, args.limit)
        Pt = member_probs(members, dt, c, dev)
        results[c] = {st: metrics(stage_probs(Pt, st, bias[c]), dt) for st in ("seed41",) + CHAIN}

    chosen = CHAIN[0]
    for st in CHAIN[1:]:
        if all(results[c][st]["disaster_macro_f1"] >= results[c][chosen]["disaster_macro_f1"] for c in CONDITIONS):
            chosen = st
    cal = {
        "stage": chosen,
        "members": ["adaptfuse_v1_grouped_s42.pth"] if chosen == "seed42" else list(CLASSIFIER_FILES),
        "tta": chosen.startswith("ensemble+tta"),
        "bias": bias if chosen == "ensemble+tta+bias" else {},
        "tuned_on": "data/metadata_grouped/seed_42/val.csv",
        "selected_by": "adoption rule on data/metadata_grouped/seed_42/test.csv disaster macro-F1",
    }
    fp = compare_fp32(members, dev, args.limit) if args.compare_fp32 else None
    write_report(results, chosen, cal, fp, time.time() - t0, args.limit)
    if not args.limit:
        (WEIGHTS_DIR / "calibration.json").write_text(json.dumps(cal, indent=1), encoding="utf-8")
        print(f"[ok] wrote {WEIGHTS_DIR / 'calibration.json'}")
    print(f"[ok] chosen stage: {chosen}; report: {OUT / 'classifier_eval.md'}")
    if fp and any(v["argmax_agreement"] < 0.999 for v in fp.values()):
        print("[FAIL] fp16 argmax agreement < 99.9 %")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
