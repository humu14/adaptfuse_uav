"""
App 2 — batch analysis (Gradio).

Upload one RGB *or* thermal video. Every frame (or every N-th frame) is analyzed;
the result is an annotated MP4 (original audio kept), a timeline plot, a summary
and a per-frame CSV/JSON export.

    python demo_app/gradio_app.py            # http://127.0.0.1:7860
"""

from __future__ import annotations

import csv
import os
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import engine  # noqa: E402  (sets KMP_DUPLICATE_LIB_OK before torch loads)
import cv2  # noqa: E402
import gradio as gr  # noqa: E402
import imageio_ffmpeg  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from engine import Analyzer, detect_modality, extract_audio, load_models, video_info  # noqa: E402
from engine.labels import DISASTER_CLASSES  # noqa: E402
from engine.render import render_frame  # noqa: E402

OUT_H = 540                       # annotated video height (panel layout assumes >= 480)
MODELS = None


def _models():
    global MODELS
    if MODELS is None:
        MODELS = load_models()
    return MODELS


def temporal_reset_gap(stride: int, fps: float) -> float:
    """Forward gap that still counts as continuous video: the spec's 2 s, or longer when the
    analysis step itself is longer (low fps or a large "every N-th frame")."""
    return max(2.0, 2.5 * int(stride) / (fps or 25.0))


def _even(x: int) -> int:
    return x - (x % 2)


def _timeline_plot(rows: list, path: Path) -> Path:
    t = [r["t"] for r in rows]
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for name in DISASTER_CLASSES:
        axes[0].plot(t, [r["disaster"]["probs"][name] for r in rows], label=name)
    axes[0].plot(t, [r["victim"]["probs"]["victim present"] for r in rows], "k--", label="victim present")
    axes[0].set_ylabel("probability"); axes[0].set_ylim(0, 1.02)
    axes[0].legend(loc="upper right", fontsize=8, ncol=3); axes[0].set_title("Scene classification")
    for k in ("rgb", "thermal", "audio"):
        axes[1].plot(t, [r["reliability"][k] for r in rows], label=k)
    axes[1].set_ylabel("RUE reliability"); axes[1].set_ylim(-0.02, 1.02); axes[1].legend(fontsize=8)
    for cls in ("fire", "smoke", "person"):
        axes[2].plot(t, [sum(b["cls"] == cls for b in r["boxes"]) for r in rows], label=cls)
    axes[2].set_ylabel("# boxes"); axes[2].set_xlabel("time (s)"); axes[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
    return path


def _summary(rows: list, info: dict, mod: dict, has_audio: bool, fps_proc: float) -> str:
    n = len(rows)
    counts = {c: sum(r["disaster"]["label"] == c for r in rows) for c in DISASTER_CLASSES}
    victim = sum(r["victim"]["index"] == 1 for r in rows)
    peak = {c: max((sum(b["cls"] == c for b in r["boxes"]) for r in rows), default=0)
            for c in ("fire", "smoke", "person")}
    tags = {}
    for r in rows:
        if r["audio"].get("status") == "ok" and r["audio"]["tags"]:
            top = r["audio"]["tags"][0]
            tags[top["name"]] = tags.get(top["name"], 0) + 1
    dis = [r["agreement"]["status"] for r in rows]
    det_only, cls_only = dis.count("detector_only"), dis.count("classifier_only")
    lines = [
        f"**Input:** {info['width']}×{info['height']}, {info['fps']:.1f} fps, {info['duration']:.1f} s — "
        f"treated as **{mod['modality'].upper()}** ({mod['reason']}); audio track: "
        f"{'yes' if has_audio else 'no'}",
        f"**Analyzed frames:** {n} at {fps_proc:.1f} frames/s processing speed",
        "",
        "| Scene label | share of frames |", "|---|---|",
        *[f"| {c} | {counts[c] / max(n, 1):.0%} |" for c in DISASTER_CLASSES],
        "",
        f"**Victim present:** {victim / max(n, 1):.0%} of frames  ",
        f"**Max boxes in one frame:** fire {peak['fire']}, smoke {peak['smoke']}, person {peak['person']}  ",
        "**Most frequent sound tag:** " + (max(tags, key=tags.get) if tags else "—") + "  ",
        f"**Scene vs. detector disagreement:** {(det_only + cls_only) / max(n, 1):.0%} of analyzed frames "
        f"(detector-only {det_only / max(n, 1):.0%}, classifier-only {cls_only / max(n, 1):.0%})",
    ]
    if mod["modality"] == "thermal":
        lines.append("\n> **Thermal-only input:** scene labels are unreliable here (grouped-test disaster "
                     "macro-F1 well below the RGB condition; see `demo_app/eval/classifier_eval.md`), and the "
                     "fire/smoke detector was trained on RGB (D-Fire).")
    return "\n".join(lines)


def analyze_video(video_path, modality_choice, stride, det_conf, person_conf, max_seconds,
                  progress=gr.Progress()):
    if not video_path:
        raise gr.Error("Upload a video first.")
    src = Path(video_path)
    m = _models()
    info = video_info(src)
    mod = detect_modality(src)
    if modality_choice != "auto":
        mod = {"modality": modality_choice, "reason": "manual override", "spread": mod["spread"]}
    audio = extract_audio(src)
    fps_src = info["fps"] or 25.0
    an = Analyzer(m, audio, mod["modality"], reset_gap=temporal_reset_gap(stride, fps_src))

    fps = info["fps"] or 25.0
    limit = int(max_seconds * fps) if max_seconds and max_seconds > 0 else None
    total = min(info["frames"], limit) if limit else info["frames"]

    work = Path(tempfile.mkdtemp(prefix="adaptfuse_"))
    out_video = work / f"{src.stem}_annotated.mp4"
    cap = cv2.VideoCapture(str(src))
    writer, rows, last, i = None, [], None, 0
    t_start = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok or (limit and i >= limit):
                break
            scale = OUT_H / frame.shape[0]
            frame = cv2.resize(frame, (_even(int(frame.shape[1] * scale)), OUT_H))
            t = i / fps
            if last is None or i % int(stride) == 0:
                last = an.analyze(frame, t, det_conf=det_conf, person_conf=person_conf)
                last["frame"] = i
                rows.append(last)
            shown = render_frame(frame, last)
            if writer is None:
                size = (_even(shown.shape[1]), _even(shown.shape[0]))
                writer = imageio_ffmpeg.write_frames(
                    str(out_video), size, fps=fps, codec="libx264", macro_block_size=1,
                    audio_path=str(src) if audio is not None else None,
                    audio_codec="aac" if audio is not None else None,
                    output_params=["-shortest"] if audio is not None else None,
                )
                writer.send(None)
            shown = cv2.cvtColor(shown[: size[1], : size[0]], cv2.COLOR_BGR2RGB)
            writer.send(np.ascontiguousarray(shown))
            i += 1
            if i % 10 == 0:
                progress(i / max(total, 1), desc=f"frame {i}/{total}")
    finally:
        cap.release()
        if writer is not None:
            writer.close()
    if not rows:
        raise gr.Error("Could not read any frames from this video.")
    fps_proc = len(rows) / (time.perf_counter() - t_start)

    json_path = work / f"{src.stem}_results.json"
    json_path.write_text(json.dumps({"video": src.name, "info": info, "modality": mod,
                                     "frames": rows}, indent=1))
    csv_path = work / f"{src.stem}_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t", "modality", "disaster", "disaster_conf", "victim", "victim_conf",
                    "nuisance", "rel_rgb", "rel_thermal", "rel_audio", "n_fire", "n_smoke", "n_person",
                    "audio_status", "audio_pipeline", "audio_top_tag", "ms",
                    "agreement", "raw_disaster", "n_raw_boxes"])
        for r in rows:
            a = r["audio"]
            w.writerow([r["frame"], r["t"], r["modality"], r["disaster"]["label"],
                        f'{r["disaster"]["conf"]:.3f}', r["victim"]["label"], f'{r["victim"]["conf"]:.3f}',
                        r["nuisance"]["label"] if r["nuisance"] else "",
                        *(f'{r["reliability"][k]:.3f}' for k in ("rgb", "thermal", "audio")),
                        *(sum(b["cls"] == c for b in r["boxes"]) for c in ("fire", "smoke", "person")),
                        a.get("status"),
                        a["pipeline"]["disaster"]["label"] if a.get("status") == "ok" else "",
                        a["tags"][0]["name"] if a.get("status") == "ok" and a["tags"] else "",
                        r["timing_ms"]["total"],
                        r["agreement"]["status"], r["raw"]["disaster"]["label"], len(r["raw_boxes"])])
    plot = _timeline_plot(rows, work / "timeline.png")
    return (str(out_video), str(plot), _summary(rows, info, mod, audio is not None, fps_proc),
            [str(csv_path), str(json_path)])


MODEL_NOTE = """
**Where each output comes from.** Scene, victim and nuisance labels and the RUE reliability
bars come from an ensemble of two **AdapFuse-V1** models (grouped split, seeds 41 + 42, flip TTA,
class calibration; see `demo_app/eval/classifier_eval.md`), trained in the AdapFuse-UAV pipeline.
The audio-only label comes from the pipeline's AudioCNN baseline. Fire and smoke boxes come from
**YOLO26s fine-tuned on CLAHE D-Fire** (test mAP@50 0.771). **Person boxes** come from the
COCO-pretrained YOLO26n, and **sound tags** from the AudioSet-pretrained PANNs Cnn6; neither was
fine-tuned in this work. Labels are smoothed over ~1 s; boxes are drawn once they persist for
2 of 3 analyzed frames; *models disagree* marks frames where scene and boxes contradict.
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="AdapFuse-UAV · batch analysis") as ui:
        gr.Markdown("# AdapFuse-UAV — video analysis\nUpload one **RGB or thermal** video "
                    "(its audio track is used when present).")
        with gr.Row():
            with gr.Column(scale=1):
                video = gr.Video(label="Input video", sources=["upload"])
                modality = gr.Radio(["auto", "rgb", "thermal"], value="auto", label="Input modality")
                stride = gr.Slider(1, 10, value=1, step=1, label="Analyze every N-th frame")
                det_conf = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="Fire/smoke confidence")
                person_conf = gr.Slider(0.05, 0.9, value=0.35, step=0.05, label="Person confidence")
                max_s = gr.Number(value=0, label="Only first N seconds (0 = whole video)")
                run = gr.Button("Analyze", variant="primary")
                samples = sorted((Path(__file__).resolve().parent / "samples").glob("*.mp4"))
                if samples:
                    gr.Examples([[str(p)] for p in samples], inputs=[video], label="Sample clips")
                gr.Markdown(MODEL_NOTE)
            with gr.Column(scale=2):
                out_video = gr.Video(label="Annotated video")
                summary = gr.Markdown()
                plot = gr.Image(label="Timeline", type="filepath")
                files = gr.File(label="Per-frame results (CSV / JSON)", file_count="multiple")
        run.click(analyze_video, [video, modality, stride, det_conf, person_conf, max_s],
                  [out_video, plot, summary, files])
    return ui


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available)")
    args = ap.parse_args()
    if args.device:
        os.environ["ADAPTFUSE_DEVICE"] = args.device
    _models()                                   # load once before the first request
    build_ui().queue().launch(server_name=args.host, server_port=args.port, show_error=True)
