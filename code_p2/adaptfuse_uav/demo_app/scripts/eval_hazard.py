"""
Measure the zero-shot hazard model (engine/hazard.py) and its fusion with the AdapFuse
scene classifier, and write weights/hazard_calibration.json.

Data: stratified subsets of the grouped split (seed_42), at most CAP images per
(source dataset, fine class): AIDER (normal / fire / collapsed building / flooded area /
traffic incident, from the folder), C2A (same classes, from the file name), FLAME
(fire / normal), FireNet (fire), SARD (normal). Optional: frames from local real-world
clips listed in REAL_CLIPS (skipped if the files are missing); they are reported only and
never used for tuning.

Tuning on val only:
  1. per-event logit bias for the zero-shot scene (coordinate search, fine macro-F1,
     with the same precision floor as eval_classifier.py)
  2. a temperature per expert (AdapFuse, zero-shot coarse), fit by val NLL
  3. fusion weight w in  p ∝ p_AdapFuse^(w/T_a) · coarse(p_zero-shot)^((1-w)/T_z): the smallest
     w whose val macro-F1 is within FUSION_TOL of the best w (prefers the zero-shot side,
     which is the one that holds up on footage unlike the training sets)

Run from demo_app/:
    python scripts/eval_hazard.py              # ~10 min on GTX 960
    python scripts/eval_hazard.py --cached     # reuse eval/hazard_probs.npz (only missing parts run)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

DEMO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO))
sys.path.insert(0, str(DEMO / "scripts"))
import engine  # noqa: E402,F401
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from engine.classifier import load_calibration  # noqa: E402
from engine.hazard import HazardModel, fuse, softmax_bias, to_coarse  # noqa: E402
from engine.labels import DISASTER_CLASSES, HAZARD_NAMES  # noqa: E402
from engine.media import extract_audio  # noqa: E402
from engine.models import CLASSIFIER_FILES, WEIGHTS_DIR, load_member  # noqa: E402
from engine.pipeline import AUDIO_WINDOW_S, _logmel, _rgb_tensor, _thermal_tensor  # noqa: E402
from eval_classifier import SPLIT_DIR, member_probs, stage_probs  # noqa: E402

OUT = DEMO / "eval"
CAP = 150
CONDITIONS = ("rgb", "thermal")
FINE = ["normal", "fire", "collapsed_building", "flooded_areas", "traffic_incident"]
FINE_OF_EVENT = {"normal": "normal", "fire": "fire", "smoke": "fire", "explosion": "fire",
                 "collapsed building": "collapsed_building", "landslide": "collapsed_building",
                 "flood": "flooded_areas", "traffic accident": "traffic_incident"}
COARSE_OF_FINE = {"normal": 0, "fire": 1, "collapsed_building": 2, "flooded_areas": 2, "traffic_incident": 3}
EVENT_FINE = np.array([FINE.index(FINE_OF_EVENT[n]) for n in HAZARD_NAMES])
PRECISION_FLOOR = 0.5
FUSION_TOL = 0.01
# Local real-world clips (demo_app/uploads/, not in git): name -> (fine label, start, end fraction)
REAL_CLIPS = {
    "fb54edf3b153": ("fire", 0.2, 1.0), "42928c7787e9": ("fire", 0.0, 1.0),
    "bd9e04a162fa": ("fire", 0.15, 1.0), "d3dd366ac602": ("fire", 0.3, 1.0),
    "69e9a09c9ce2": ("fire", 0.0, 1.0), "f04359e13141": ("fire", 0.0, 1.0),
    "f829f6eef594": ("flooded_areas", 0.0, 1.0), "02fa4278b67d": ("flooded_areas", 0.85, 1.0),
    "0cab2631ba44": ("collapsed_building", 0.0, 1.0),
}
FRAMES_PER_CLIP = 8


def fine_label(row) -> str | None:
    src, p = row["source_dataset"], str(row["rgb_path"])
    if src == "aider":
        return Path(p).parent.name
    if src == "c2a":
        k = Path(p).name.rsplit("_image", 1)[0]
        return {"flood": "flooded_areas", "fire": "fire", "collapsed_building": "collapsed_building",
                "traffic_incident": "traffic_incident"}.get(k)
    if src == "flame":
        return "fire" if int(row["disaster_label"]) == 1 else "normal"
    return {"firenet": "fire", "sard": "normal"}.get(src)


def load_subset(split: str) -> pd.DataFrame:
    df = pd.read_csv(SPLIT_DIR / f"{split}.csv")
    df = df[(df.has_rgb == 1) & (df.has_thermal == 1)].copy()
    df["fine"] = [fine_label(r) for _, r in df.iterrows()]
    df = df[df.fine.notna()]
    df = df.groupby(["source_dataset", "fine"], group_keys=False).apply(
        lambda g: g.sample(min(len(g), CAP), random_state=0))
    return df.reset_index(drop=True)


def real_frames() -> list:
    out = []
    for name, (fine, a, b) in REAL_CLIPS.items():
        path = DEMO / "uploads" / f"{name}.mp4"
        if not path.exists():
            continue
        audio = extract_audio(path)
        cap = cv2.VideoCapture(str(path))
        n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS) or 25.0
        for k in range(FRAMES_PER_CLIP):
            i = int(n * (a + (b - a) * (k + 0.5) / FRAMES_PER_CLIP))
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, fr = cap.read()
            if ok:
                out.append({"clip": name, "fine": fine, "frame": fr, "t": i / fps, "audio": audio})
        cap.release()
    return out


def hazard_logits(hz: HazardModel, images: list) -> np.ndarray:
    out = []
    for i, im in enumerate(images):
        out.append(hz.logits(im, dense=False)[0])
        if i % 50 == 0:
            print(f"\r  zero-shot {i + 1}/{len(images)}", end="", flush=True)
    print()
    return np.stack(out)


def adaptfuse_real(members, frames, condition, dev, bias: dict) -> np.ndarray:
    from engine.classifier import SceneClassifier
    clf = SceneClassifier(list(members.values()), dev, tta=True, bias=bias)
    out = []
    for f in frames:
        if f["audio"] is not None:
            y = f["audio"].window(f["t"], AUDIO_WINDOW_S, 16000)
            au, ha = (_logmel(y), 1.0) if np.sqrt(np.mean(y ** 2)) >= 1e-4 else (torch.zeros(1, 1, 64, 63), 0.0)
        else:
            au, ha = torch.zeros(1, 1, 64, 63), 0.0
        if condition == "rgb":
            r, t, has = _rgb_tensor(f["frame"]), torch.zeros(1, 1, 224, 224), (1.0, 0.0, ha)
        else:
            r, t, has = torch.zeros(1, 3, 224, 224), _thermal_tensor(f["frame"]), (0.0, 1.0, ha)
        out.append(clf.predict(r, t, au, has, condition)["disaster"])
    return np.stack(out)


def to_thermal(bgr):
    return cv2.cvtColor(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


def compute(dev, data: dict | None = None) -> dict:
    hz = HazardModel(WEIGHTS_DIR / "hazard_clip", WEIGHTS_DIR / "hazard_text.pt", dev)
    cal = load_calibration(WEIGHTS_DIR / "calibration.json")
    members = {n: load_member(WEIGHTS_DIR / n, dev) for n in CLASSIFIER_FILES}
    data = data or {}
    for split in ("val", "test"):
        if f"{split}_thermal_af" in data:
            continue
        df = load_subset(split)
        print(f"[{split}] {len(df)} rows: " + ", ".join(f"{k}={v}" for k, v in df.fine.value_counts().items()))
        data[f"{split}_fine"] = np.array([FINE.index(f) for f in df.fine])
        data[f"{split}_src"] = df.source_dataset.to_numpy()
        for c in CONDITIONS:
            col = "rgb_path" if c == "rgb" else "thermal_path"
            ims = [cv2.imread(p, cv2.IMREAD_COLOR) for p in df[col]]
            data[f"{split}_{c}_zs"] = hazard_logits(hz, ims)
            P = member_probs(members, df, c, dev)
            data[f"{split}_{c}_af"] = stage_probs(P, "ensemble+tta+bias", cal["bias"].get(c))["disaster"]
        OUT.mkdir(exist_ok=True)
        np.savez_compressed(OUT / "hazard_probs.npz", **data)          # resumable if a later step fails
    rf = real_frames()
    if rf:
        print(f"[real] {len(rf)} frames from {len({f['clip'] for f in rf})} clips")
        data["real_fine"] = np.array([FINE.index(f["fine"]) for f in rf])
        data["real_clip"] = np.array([f["clip"] for f in rf])
        for c in CONDITIONS:
            ims = [f["frame"] if c == "rgb" else to_thermal(f["frame"]) for f in rf]
            data[f"real_{c}_zs"] = hazard_logits(hz, ims)
            fr = [{**f, "frame": im} for f, im in zip(rf, ims)]
            data[f"real_{c}_af"] = adaptfuse_real(members, fr, c, dev, cal["bias"])
    return data


# ── metrics / tuning ──────────────────────────────────────────────────────────

def fine_pred(p_event: np.ndarray) -> np.ndarray:
    return EVENT_FINE[p_event.argmax(1)]


def coarse_y(fine_idx: np.ndarray) -> np.ndarray:
    return np.array([COARSE_OF_FINE[FINE[i]] for i in fine_idx])


def mf1(y, p, labels) -> float:
    return float(f1_score(y, p, labels=labels, average="macro", zero_division=0))


def tune_event_bias(logits: np.ndarray, y_fine: np.ndarray) -> list:
    E = logits.shape[1]
    b = np.zeros(E)
    grid = np.round(np.arange(-3.0, 3.0001, 0.25), 2)
    base = fine_pred(softmax_bias(logits, None))
    base_n = [int((base == c).sum()) for c in range(len(FINE))]
    base_prec = [((base == c) & (y_fine == c)).sum() / base_n[c] if base_n[c] else 1.0 for c in range(len(FINE))]

    def score(bb):
        pred = fine_pred(softmax_bias(logits, bb))
        for c in range(len(FINE)):
            n = int((pred == c).sum())
            if not n:
                continue
            prec = ((pred == c) & (y_fine == c)).sum() / n
            if prec < PRECISION_FLOOR and (base_prec[c] >= PRECISION_FLOOR or n > base_n[c] or prec < base_prec[c]):
                return -1.0
        return mf1(y_fine, pred, list(range(len(FINE))))

    best = score(b)
    for _ in range(3):
        for e in range(1, E):
            for g in grid:
                t = b.copy(); t[e] = g
                s = score(t)
                if s > best + 1e-9:
                    best, b = s, t
    return [float(x) for x in b]


def fit_temperature(p: np.ndarray, y: np.ndarray) -> float:
    """Temperature T minimising val NLL of softmax(log p / T)."""
    lp = np.log(np.clip(p, 1e-8, 1.0))
    best = (1.0, np.inf)
    for T in np.round(np.arange(0.25, 8.0001, 0.05), 2):
        q = softmax_bias(lp / T, None)
        nll = -np.mean(np.log(np.clip(q[np.arange(len(y)), y], 1e-8, 1.0)))
        if nll < best[1]:
            best = (float(T), nll)
    return best[0]


def evaluate(data: dict) -> tuple[dict, dict]:
    cal = {"bias": {}, "fusion_weight": {}, "temperature": {}, "zone": {"gate": 0.25, "thr": 0.5, "min_frac": 0.02}}
    report = {}
    weights = np.round(np.arange(0, 1.0001, 0.05), 2)
    for c in CONDITIONS:
        yv, yt = data["val_fine"], data["test_fine"]
        bias = tune_event_bias(data[f"val_{c}_zs"], yv)
        cal["bias"][c] = bias
        zs = {s: softmax_bias(data[f"{s}_{c}_zs"], bias) for s in ("val", "test", "real") if f"{s}_{c}_zs" in data}
        zs_raw = {s: softmax_bias(data[f"{s}_{c}_zs"], None) for s in zs}
        ycv = coarse_y(yv)
        T = {"adaptfuse": fit_temperature(data[f"val_{c}_af"], ycv), "zero_shot": fit_temperature(to_coarse(zs["val"]), ycv)}
        cal["temperature"][c] = T

        def fused(p_af, p_ev, w):
            return fuse(p_af, p_ev, w, T["adaptfuse"], T["zero_shot"])
        curve = {}
        for w in weights:
            curve[float(w)] = {s: mf1(coarse_y(data[f"{s}_fine"]), fused(data[f"{s}_{c}_af"], zs[s], w).argmax(1),
                                      [0, 1, 2, 3]) for s in zs}
        best = max(v["val"] for v in curve.values())
        w_sel = min(w for w, v in curve.items() if v["val"] >= best - FUSION_TOL)
        cal["fusion_weight"][c] = w_sel

        rows = {}
        for s in zs:
            yc = coarse_y(data[f"{s}_fine"])
            af = data[f"{s}_{c}_af"]
            rows[s] = {
                "AdapFuse ensemble (app before)": {"coarse_f1": mf1(yc, af.argmax(1), [0, 1, 2, 3]),
                                                    "coarse_acc": float((af.argmax(1) == yc).mean())},
                "zero-shot, no bias": {"coarse_f1": mf1(yc, to_coarse(zs_raw[s]).argmax(1), [0, 1, 2, 3]),
                                       "coarse_acc": float((to_coarse(zs_raw[s]).argmax(1) == yc).mean()),
                                       "fine_f1": mf1(data[f"{s}_fine"], fine_pred(zs_raw[s]), list(range(5)))},
                "zero-shot + val bias": {"coarse_f1": mf1(yc, to_coarse(zs[s]).argmax(1), [0, 1, 2, 3]),
                                         "coarse_acc": float((to_coarse(zs[s]).argmax(1) == yc).mean()),
                                         "fine_f1": mf1(data[f"{s}_fine"], fine_pred(zs[s]), list(range(5)))},
                f"fused w={w_sel:.2f} (app now)": {
                    "coarse_f1": mf1(yc, fused(af, zs[s], w_sel).argmax(1), [0, 1, 2, 3]),
                    "coarse_acc": float((fused(af, zs[s], w_sel).argmax(1) == yc).mean())},
            }
            if s == "test":
                pf = fine_pred(zs[s])
                rows[s]["per_fine_f1"] = {FINE[k]: float(f1_score(data["test_fine"], pf, labels=[k], average="macro",
                                                                  zero_division=0)) for k in range(5)}
        if "real" in zs:
            per_clip = {}
            fz = fused(data[f"real_{c}_af"], zs["real"], w_sel)
            for clip in dict.fromkeys(data["real_clip"]):
                idx = data["real_clip"] == clip
                y = COARSE_OF_FINE[FINE[data["real_fine"][idx][0]]]
                per_clip[clip] = {
                    "truth": FINE[data["real_fine"][idx][0]],
                    "adaptfuse_acc": float((data[f"real_{c}_af"][idx].argmax(1) == y).mean()),
                    "fused_acc": float((fz[idx].argmax(1) == y).mean()),
                    "events": [HAZARD_NAMES[i] for i in zs["real"][idx].argmax(1)]}
            rows["real_per_clip"] = per_clip
        report[c] = {"rows": rows, "curve": curve, "bias": bias, "fusion_weight": w_sel, "temperature": T}
    return report, cal


def write(report: dict, cal: dict, n: dict, secs: float) -> None:
    OUT.mkdir(exist_ok=True)
    (OUT / "hazard_eval.json").write_text(json.dumps({"report": report, "calibration": cal, "n": n}, indent=1))
    L = ["# Zero-shot hazard model and fused scene (grouped split, seed_42)", "",
         f"Stratified subsets, at most {CAP} images per (dataset, class): val n = {n['val']}, test n = {n['test']}"
         + (f", real-world clip frames n = {n['real']}" if n.get("real") else "") + ".",
         "Event bias and fusion weight are tuned on **val**; test and real-clip numbers are reported only.",
         "Coarse = the 4 scene classes (" + ", ".join(DISASTER_CLASSES) + "); fine = normal / fire / collapsed "
         "building / flooded area / traffic incident.",
         "Fused = p ∝ p_AdapFuse^(w/T_a) · coarse(p_zero-shot)^((1-w)/T_z), temperatures fit by val NLL.", ""]
    for c in CONDITIONS:
        r = report[c]["rows"]
        L += [f"## Condition: {c}", "", "| Model | split | coarse macro-F1 | coarse acc | fine macro-F1 |",
              "|---|---|---|---|---|"]
        for s in ("val", "test", "real"):
            if s not in r:
                continue
            for name, m in r[s].items():
                if name == "per_fine_f1":
                    continue
                fine = f"{m['fine_f1']:.3f}" if "fine_f1" in m else "—"
                L.append(f"| {name} | {s} | {m['coarse_f1']:.3f} | {m['coarse_acc']:.3f} | {fine} |")
        L += ["", "Test F1 per fine class (zero-shot + val bias): " + ", ".join(
            f"{k} {v:.3f}" for k, v in r["test"]["per_fine_f1"].items()), ""]
        if "real_per_clip" in r:
            L += ["Real-world clips (frame accuracy on the coarse class):", "",
                  "| clip | truth | AdapFuse | fused | zero-shot events |", "|---|---|---|---|---|"]
            for clip, v in r["real_per_clip"].items():
                ev = ", ".join(f"{k}×{v['events'].count(k)}" for k in dict.fromkeys(v["events"]))
                L.append(f"| {clip} | {v['truth']} | {v['adaptfuse_acc']:.2f} | {v['fused_acc']:.2f} | {ev} |")
            L.append("")
        L += ["Fusion weight curve (coarse macro-F1):", "", "| w (AdapFuse) | " + " | ".join(
            s for s in ("val", "test", "real") if s in r) + " |", "|---|" + "---|" * len([s for s in ("val", "test", "real") if s in r])]
        for w, v in report[c]["curve"].items():
            mark = " ←" if abs(w - report[c]["fusion_weight"]) < 1e-9 else ""
            L.append(f"| {w:.2f}{mark} | " + " | ".join(f"{v[s]:.3f}" for s in ("val", "test", "real") if s in v) + " |")
        L.append("")
    L += ["## Calibration written to `weights/hazard_calibration.json`", "", "```json",
          json.dumps(cal, indent=1), "```", "", f"Run time: {secs / 60:.1f} min."]
    (OUT / "hazard_eval.md").write_text("\n".join(L), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--cached", action="store_true", help="reuse eval/hazard_probs.npz")
    args = ap.parse_args()
    t0 = time.time()
    cache = OUT / "hazard_probs.npz"
    data = dict(np.load(cache, allow_pickle=True)) if args.cached and cache.exists() else {}
    if "test_thermal_af" not in data or "real_thermal_af" not in data:
        data = compute(torch.device(args.device), data)
        np.savez_compressed(cache, **data)
    report, cal = evaluate(data)
    cal["tuned_on"] = "data/metadata_grouped/seed_42/val.csv (stratified subset)"
    n = {s: int(len(data[f"{s}_fine"])) for s in ("val", "test", "real") if f"{s}_fine" in data}
    write(report, cal, n, time.time() - t0)
    (WEIGHTS_DIR / "hazard_calibration.json").write_text(json.dumps(cal, indent=1), encoding="utf-8")
    for c in CONDITIONS:
        print(c, json.dumps({s: {k: round(v["coarse_f1"], 3) for k, v in r.items() if isinstance(v, dict) and "coarse_f1" in v}
                             for s, r in report[c]["rows"].items() if s in ("val", "test", "real")}, indent=1))
        print(c, "fusion weight", cal["fusion_weight"][c])
    print(f"[ok] {OUT / 'hazard_eval.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
