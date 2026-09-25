"""
Temporal consistency for the demo: time-based EMA of class probabilities, box
persistence (k-of-n frames), and a scene-vs-detector agreement flag. Nothing here
overrides a model output; it only smooths and annotates.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

FIRE_SCENE = 1                      # index of "fire / smoke" in DISASTER_CLASSES
HAZARD_BOXES = ("fire", "smoke")


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def ema_alpha(dt: float, tau: float) -> float:
    return 1.0 - math.exp(-max(dt, 0.0) / tau)


class TemporalState:
    def __init__(self, tau: float = 0.5, window: int = 3, min_hits: int = 2, iou_thr: float = 0.3,
                 reset_gap: float = 2.0, disagree_conf: float = 0.5, classifier_only_s: float = 2.0,
                 scene_names: list | None = None):
        self.tau, self.window, self.min_hits, self.iou_thr = tau, window, min_hits, iou_thr
        self.reset_gap, self.disagree_conf, self.classifier_only_s = reset_gap, disagree_conf, classifier_only_s
        self.scene_names = scene_names
        self.reset()

    def reset(self) -> None:
        self.last_t = None
        self.probs = None
        self.history = deque(maxlen=self.window)
        self.fire_scene_since = None
        self.last_hazard_box_t = None
        self.status, self.status_since = "agree", None

    def update(self, t: float, raw: dict, boxes: list):
        if self.last_t is not None and (t < self.last_t - 1e-6 or t - self.last_t > self.reset_gap):
            self.reset()
        cur = {k: np.asarray(v, dtype=np.float32) for k, v in raw.items()}
        if self.probs is None:
            self.probs = {k: v.copy() for k, v in cur.items()}
        else:
            a = ema_alpha(t - self.last_t, self.tau)
            self.probs = {k: self.probs[k] + a * (cur[k] - self.probs[k]) for k in cur}
        self.last_t = t

        self.history.append(boxes)
        past_frames = list(self.history)[:-1]
        confirmed = []
        for b in boxes:
            matches = [b]
            for past in past_frames:
                same = [p for p in past if p["cls"] == b["cls"]]
                if not same:
                    continue
                best = max(same, key=lambda p: iou(p["box"], b["box"]))
                if iou(best["box"], b["box"]) >= self.iou_thr:
                    matches.append(best)
            if len(matches) >= self.min_hits:
                confirmed.append({**b, "conf": float(np.mean([m["conf"] for m in matches])),
                                  "hits": len(matches)})
        return {k: v.copy() for k, v in self.probs.items()}, confirmed, self._agreement(t, confirmed)

    def _agreement(self, t: float, confirmed: list) -> dict:
        scene = int(np.argmax(self.probs["disaster"]))
        hazard = [b for b in confirmed if b["cls"] in HAZARD_BOXES]
        if hazard:
            self.last_hazard_box_t = t
        if scene == FIRE_SCENE:
            if self.fire_scene_since is None:
                self.fire_scene_since = t
        else:
            self.fire_scene_since = None

        status, reason = "agree", ""
        strong = [b for b in hazard if b["conf"] >= self.disagree_conf]
        if strong and scene != FIRE_SCENE:
            top = max(strong, key=lambda b: b["conf"])
            reason = f'detector: {top["cls"]} {top["conf"]:.2f}'
            if self.scene_names:
                reason += f" · scene: {self.scene_names[scene]}"
            status = "detector_only"
        elif scene == FIRE_SCENE:
            ref = self.fire_scene_since
            if self.last_hazard_box_t is not None:
                ref = max(ref, self.last_hazard_box_t)
            if t - ref >= self.classifier_only_s - 1e-9:
                status = "classifier_only"
                reason = f"scene: fire / smoke for {t - ref:.1f} s · detector: no fire/smoke box"

        if status != self.status or self.status_since is None:
            self.status, self.status_since = status, t
        return {"status": status, "since": self.status_since, "reason": reason}
