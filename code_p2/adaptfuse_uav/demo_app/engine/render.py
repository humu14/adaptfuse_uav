"""Burn analysis results into a frame (used for the Gradio annotated video)."""

from __future__ import annotations

import cv2
import numpy as np

# BGR
COLORS = {"fire": (40, 60, 235), "smoke": (170, 170, 170), "person": (60, 200, 60)}
PANEL_BG = (24, 24, 24)
TEXT = (240, 240, 240)
MUTED = (160, 160, 160)
WARN = (32, 176, 255)  # BGR amber
MOD_COLORS = {"rgb": (230, 160, 60), "thermal": (60, 120, 240), "audio": (180, 90, 200)}


def _text(img, s, org, scale=0.5, color=TEXT, thick=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_boxes(frame: np.ndarray, boxes: list) -> None:
    h, w = frame.shape[:2]
    for b in boxes:
        x1, y1, x2, y2 = (int(b["box"][0] * w), int(b["box"][1] * h),
                          int(b["box"][2] * w), int(b["box"][3] * h))
        c = COLORS.get(b["cls"], (255, 255, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
        label = f'{b["cls"]} {b["conf"]:.2f}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), c, -1)
        _text(frame, label, (x1 + 3, max(th, y1 - 4)), 0.5, (0, 0, 0) if b["cls"] == "smoke" else TEXT)


def _bar(img, x, y, w, frac, color, label, value):
    cv2.rectangle(img, (x, y), (x + w, y + 10), (60, 60, 60), -1)
    cv2.rectangle(img, (x, y), (x + int(w * max(0.0, min(1.0, frac))), y + 10), color, -1)
    _text(img, label, (x, y - 4), 0.42, MUTED)
    _text(img, value, (x + w + 6, y + 10), 0.42, TEXT)


def render_frame(frame: np.ndarray, res: dict, panel_w: int = 300) -> np.ndarray:
    """Return frame with boxes drawn and a side panel of labels (width + panel_w)."""
    frame = frame.copy()
    draw_boxes(frame, res["boxes"])
    h = frame.shape[0]
    panel = np.full((h, panel_w, 3), PANEL_BG, dtype=np.uint8)
    x, y = 12, 24
    _text(panel, f'AdapFuse-UAV  t={res["t"]:.1f}s', (x, y), 0.55, TEXT, 1); y += 22
    _text(panel, f'input: {res["modality"].upper()}', (x, y), 0.45, MUTED); y += 26

    d, v, n = res["disaster"], res["victim"], res["nuisance"]
    _text(panel, "SCENE", (x, y), 0.42, MUTED); y += 20
    _text(panel, f'{d["label"]}  {d["conf"]:.2f}', (x, y), 0.6, TEXT, 2); y += 22
    _text(panel, f'{v["label"]}  {v["conf"]:.2f}', (x, y), 0.5); y += 20
    if n:
        _text(panel, f'nuisance: {n["label"]} {n["conf"]:.2f}', (x, y), 0.45); y += 26
    ag = res.get("agreement", {"status": "agree"})
    if ag["status"] != "agree":
        _text(panel, "MODELS DISAGREE", (x, y), 0.5, WARN, 2); y += 18
        for part in ag["reason"].split(" · ")[:2]:
            _text(panel, part[:36], (x, y), 0.42, WARN); y += 16
        y += 6

    _text(panel, "RUE RELIABILITY", (x, y), 0.42, MUTED); y += 18
    for k in ("rgb", "thermal", "audio"):
        _bar(panel, x, y + 12, panel_w - 80, res["reliability"][k], MOD_COLORS[k], k,
             f'{res["reliability"][k]:.2f}')
        y += 30
    y += 6

    a = res["audio"]
    _text(panel, "AUDIO", (x, y), 0.42, MUTED); y += 20
    if a.get("status") == "ok":
        ad = a["pipeline"]["disaster"]
        _text(panel, f'pipeline: {ad["label"]} {ad["conf"]:.2f}', (x, y), 0.45); y += 18
        for tag in a["tags"][:3]:
            _bar(panel, x, y + 12, panel_w - 80, tag["score"], MOD_COLORS["audio"],
                 tag["name"][:30], f'{tag["score"]:.2f}')
            y += 30
    else:
        _text(panel, a.get("status", "none"), (x, y), 0.45); y += 20

    y = h - 34
    counts = {}
    for b in res["boxes"]:
        counts[b["cls"]] = counts.get(b["cls"], 0) + 1
    _text(panel, "boxes: " + (", ".join(f"{k} x{c}" for k, c in counts.items()) or "none"),
          (x, y), 0.45)
    _text(panel, f'{res["timing_ms"]["total"]:.0f} ms/frame', (x, y + 20), 0.42, MUTED)
    return np.hstack([frame, panel])
