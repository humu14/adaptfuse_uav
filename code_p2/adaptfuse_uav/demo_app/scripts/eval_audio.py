"""
Compare AudioSet taggers for the app's sound events (engine/labels.py SOUND_GROUPS) on
ESC-50, and write per-group thresholds to weights/sound_thresholds.json.

ESC-50 classes are mapped to the sound groups below; classes without a group are
negatives (they must not raise a hazard group), clock_alarm / water_drops are left out
as ambiguous. Thresholds are tuned on folds 1-3 (best F1 per group), reported on folds 4-5.
Windows: the loudest W seconds of each 5 s clip (the app slides a window over the track,
so some window contains the event).

    python scripts/eval_audio.py --taggers cnn6 cnn14_16k [--ast-dir PATH]
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
import librosa  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from engine.labels import SOUND_GROUPS  # noqa: E402
from engine.models import PROJECT_DIR, WEIGHTS_DIR, load_tagger  # noqa: E402

ESC = PROJECT_DIR / "datasets/raw/ESC50/ESC-50-master"
OUT = DEMO / "eval"
ESC_TO_GROUP = {
    "crackling_fire": "fire crackling", "fireworks": "explosion / blast", "siren": "siren / alarm",
    "crying_baby": "screaming / crying", "glass_breaking": "crash / collapse",
    "sea_waves": "rushing water", "pouring_water": "rushing water", "toilet_flush": "rushing water",
    "rain": "rain / thunder", "thunderstorm": "rain / thunder", "helicopter": "helicopter / drone",
    "airplane": "helicopter / drone", "engine": "vehicles / traffic", "car_horn": "vehicles / traffic",
    "train": "vehicles / traffic", "wind": "wind",
}
SKIP = {"clock_alarm", "water_drops"}
GROUPS = list(SOUND_GROUPS)
_WAV: dict = {}
GIDX = [SOUND_GROUPS[g]["idx"] for g in GROUPS]


def loudest(y: np.ndarray, sr: int, seconds: float) -> np.ndarray:
    n = int(seconds * sr)
    if len(y) <= n:
        return np.pad(y, (0, n - len(y)))
    c = np.concatenate([[0.0], np.cumsum(y.astype(np.float64) ** 2)])
    starts = np.arange(0, len(y) - n + 1, sr // 10)
    s = int(starts[np.argmax(c[starts + n] - c[starts])])
    return y[s:s + n]


class Tagger:
    def __init__(self, kind: str, dev, ast_dir: str | None = None):
        self.kind, self.dev = kind, dev
        if kind in ("cnn6", "cnn14_16k"):
            self.m = load_tagger(kind)
            self.sr = self.m.SAMPLE_RATE
        elif kind == "ast":
            from transformers import ASTFeatureExtractor, ASTForAudioClassification
            self.sr = 16000
            self.fe = ASTFeatureExtractor.from_pretrained(ast_dir)
            self.m = ASTForAudioClassification.from_pretrained(ast_dir)
        self.m.to(dev).eval()

    @torch.no_grad()
    def __call__(self, y: np.ndarray) -> np.ndarray:
        if self.kind == "ast":
            x = self.fe(y, sampling_rate=16000, return_tensors="pt")["input_values"].to(self.dev)
            return torch.sigmoid(self.m(input_values=x).logits)[0].cpu().numpy()
        return self.m(torch.from_numpy(y.astype(np.float32))[None].to(self.dev))[0].cpu().numpy()


def group_scores(tags: np.ndarray) -> np.ndarray:
    return np.array([tags[i].max() for i in GIDX])


def best_threshold(s: np.ndarray, pos: np.ndarray) -> tuple[float, float]:
    best = (0.5, -1.0)
    for t in np.round(np.arange(0.02, 0.81, 0.01), 2):
        pred = s >= t
        tp, fp, fn = (pred & pos).sum(), (pred & ~pos).sum(), (~pred & pos).sum()
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        if f1 > best[1]:
            best = (float(t), float(f1))
    return best


def run(kind: str, window: float, dev, ast_dir) -> dict:
    import csv
    meta = [r for r in csv.DictReader(open(ESC / "meta/esc50.csv", encoding="utf-8")) if r["category"] not in SKIP]
    tagger = Tagger(kind, dev, ast_dir)
    S, folds, truth = [], [], []
    t0 = time.perf_counter()
    for r in meta:
        key = (r["filename"], tagger.sr)
        if key not in _WAV:
            _WAV[key] = librosa.load(ESC / "audio" / r["filename"], sr=tagger.sr, mono=True)[0]
        y = _WAV[key]
        S.append(group_scores(tagger(loudest(y, tagger.sr, window))))
        folds.append(int(r["fold"])); truth.append(ESC_TO_GROUP.get(r["category"]))
    ms = (time.perf_counter() - t0) / len(meta) * 1000
    S, folds = np.stack(S), np.array(folds)
    tune, test = folds <= 3, folds >= 4
    res = {"tagger": kind, "window_s": window, "ms_per_clip_incl_io": round(ms, 1), "groups": {}}
    for gi, g in enumerate(GROUPS):
        pos = np.array([t == g for t in truth])
        if not pos.any():
            continue
        thr, f1_tune = best_threshold(S[tune, gi], pos[tune])
        pred = S[test, gi] >= thr
        p = pos[test]
        tp, fp, fn = (pred & p).sum(), (pred & ~p).sum(), (~pred & p).sum()
        res["groups"][g] = {"thr": thr, "test_f1": float(2 * tp / max(1, 2 * tp + fp + fn)),
                            "test_recall": float(tp / max(1, p.sum())), "test_precision": float(tp / max(1, tp + fp)),
                            "n_pos_test": int(p.sum())}
    mapped = test & np.array([t is not None for t in truth])
    top = [GROUPS[i] for i in S[mapped].argmax(1)]
    res["top_group_acc_test"] = float(np.mean([a == b for a, b in zip(top, np.array(truth, dtype=object)[mapped])]))
    res["mean_group_f1_test"] = float(np.mean([v["test_f1"] for v in res["groups"].values()]))
    hz = [i for i, g in enumerate(GROUPS) if SOUND_GROUPS[g]["hazard"]]
    neg = test & np.array([t is None for t in truth])
    thr = np.array([res["groups"].get(GROUPS[i], {"thr": 0.3})["thr"] for i in hz])
    res["hazard_false_alarm_rate_test"] = float((S[neg][:, hz] >= thr).any(1).mean())
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--taggers", nargs="+", default=["cnn6", "cnn14_16k"])
    ap.add_argument("--windows", nargs="+", type=float, default=[2.0, 3.0, 5.0])
    ap.add_argument("--ast-dir", default=None)
    ap.add_argument("--select", default=None, help="tagger:window whose thresholds go to weights/")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for k in args.taggers:
        for w in args.windows:
            r = run(k, w, dev, args.ast_dir)
            print(json.dumps({x: r[x] for x in r if x != "groups"}))
            results.append(r)
    OUT.mkdir(exist_ok=True)
    (OUT / "audio_eval.json").write_text(json.dumps(results, indent=1))
    L = ["# Sound-event taggers on ESC-50", "",
         "ESC-50 classes mapped to the app's sound groups (others are negatives). Per-group thresholds "
         "tuned on folds 1-3 (best F1), numbers below on folds 4-5. Window = loudest W s of each clip.", "",
         "| Tagger | window | mean group F1 | top-group acc | hazard false-alarm rate | ms / clip |",
         "|---|---|---|---|---|---|"]
    for r in results:
        L.append(f"| {r['tagger']} | {r['window_s']:.0f} s | {r['mean_group_f1_test']:.3f} | "
                 f"{r['top_group_acc_test']:.3f} | {r['hazard_false_alarm_rate_test']:.3f} | {r['ms_per_clip_incl_io']} |")
    L += ["", "Per-group test F1 (threshold):", "", "| Group | " + " | ".join(
        f"{r['tagger']} {r['window_s']:.0f}s" for r in results) + " |", "|---|" + "---|" * len(results)]
    for g in GROUPS:
        if g in results[0]["groups"]:
            L.append(f"| {g} | " + " | ".join(
                f"{r['groups'][g]['test_f1']:.2f} ({r['groups'][g]['thr']:.2f})" for r in results) + " |")
    (OUT / "audio_eval.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    if args.select:
        k, w = args.select.split(":")
        r = next(r for r in results if r["tagger"] == k and abs(r["window_s"] - float(w)) < 1e-6)
        thr = {g: v["thr"] for g, v in r["groups"].items()}
        (WEIGHTS_DIR / "sound_thresholds.json").write_text(json.dumps(
            {"tagger": k, "window_s": float(w), "thresholds": thr,
             "tuned_on": "ESC-50 folds 1-3 (scripts/eval_audio.py)"}, indent=1), encoding="utf-8")
        print(f"[ok] wrote weights/sound_thresholds.json for {k} {w}s")
    print(f"[ok] {OUT / 'audio_eval.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
