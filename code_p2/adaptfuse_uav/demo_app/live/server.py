"""
App 1 — real-time analysis (FastAPI + WebSocket).

The browser plays the uploaded video with its sound. While it plays, the page
grabs the frame on screen, sends it with its timestamp over a WebSocket, and
draws the returned boxes and labels on top of the video. The server keeps the
video's audio track and cuts the 2 s window ending at that timestamp, so the
audio labels stay in sync with what the viewer hears.

    python demo_app/live/server.py            # http://127.0.0.1:8000
"""

from __future__ import annotations

import asyncio
import base64
import os
import shutil
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO_DIR))

import engine  # noqa: E402,F401  (sets KMP_DUPLICATE_LIB_OK before torch loads)
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from starlette.concurrency import run_in_threadpool  # noqa: E402

from engine import Analyzer, detect_modality, extract_audio, load_models, video_info  # noqa: E402
from engine.media import AudioTrack, transcode_to_h264  # noqa: E402
from engine.modality import THERMAL_NAME_HINTS  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
UPLOADS = DEMO_DIR / "uploads"
SAMPLES = DEMO_DIR / "samples"
BROWSER_OK = {".mp4", ".webm", ".m4v"}
MAX_UPLOAD_MB = 500

MODELS = None
VIDEOS: dict[str, dict] = {}          # id -> {path, info, modality, audio}


def _load() -> None:
    global MODELS
    UPLOADS.mkdir(exist_ok=True)
    MODELS = load_models()
    # warm-up so the first real frame is not slow
    dummy = np.zeros((360, 640, 3), np.uint8)
    tone = AudioTrack(*(0.1 * np.sin(2 * np.pi * 440 * np.arange(0, 3, 1 / sr)).astype(np.float32)
                        for sr in (16000, 32000)))
    Analyzer(MODELS, tone, "rgb").analyze(dummy, 2.0)       # warms detectors, classifier and audio nets
    Analyzer(MODELS, None, "thermal").analyze(dummy, 0.0)
    print(f"[live] models ready on {MODELS.device}")


@asynccontextmanager
async def lifespan(_app):
    await run_in_threadpool(_load)
    yield


app = FastAPI(title="AdapFuse-UAV live demo", lifespan=lifespan)


def _register(path: Path, display_name: str) -> dict:
    info = video_info(path)
    if path.suffix.lower() not in BROWSER_OK:
        mp4 = path.with_suffix(".mp4")
        transcode_to_h264(path, mp4)
        path = mp4
    vid = uuid.uuid4().hex[:12]
    # Uploads are stored under random names, so check the original name for hints.
    if any(k in Path(display_name).stem.lower() for k in THERMAL_NAME_HINTS):
        mod = {"modality": "thermal", "reason": "file name", "spread": None}
    else:
        mod = detect_modality(path)
    audio = extract_audio(path)
    VIDEOS[vid] = {"path": path, "info": info, "modality": mod, "audio": audio, "name": display_name}
    return {"id": vid, "url": f"/media/{vid}", "name": display_name, "info": info,
            "modality": mod, "has_audio": audio is not None,
            "device": str(MODELS.device) if MODELS else "?"}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    suffix = Path(file.filename or "video.mp4").suffix.lower() or ".mp4"
    dst = UPLOADS / f"{uuid.uuid4().hex[:12]}{suffix}"
    size = 0
    with open(dst, "wb") as f:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD_MB * (1 << 20):
                f.close(); dst.unlink(missing_ok=True)
                raise HTTPException(413, f"File larger than {MAX_UPLOAD_MB} MB")
            f.write(chunk)
    try:
        return await run_in_threadpool(_register, dst, file.filename or dst.name)
    except Exception as e:  # unreadable / not a video
        dst.unlink(missing_ok=True)
        print(f"[live] rejected upload {file.filename!r}: {e}")
        raise HTTPException(400, "Could not read this file as a video.")


@app.get("/api/samples")
def samples():
    return [p.name for p in sorted(SAMPLES.glob("*.mp4"))] if SAMPLES.exists() else []


@app.post("/api/samples/{name}")
async def use_sample(name: str):
    src = SAMPLES / Path(name).name
    if not src.exists():
        raise HTTPException(404, "sample not found")
    dst = UPLOADS / f"{uuid.uuid4().hex[:12]}_{src.name}"
    shutil.copy2(src, dst)
    return await run_in_threadpool(_register, dst, src.name)


@app.get("/media/{vid}")
def media(vid: str):
    v = VIDEOS.get(vid)
    if not v:
        raise HTTPException(404)
    return FileResponse(v["path"], media_type="video/mp4")


@app.get("/api/health")
def health():
    return {"ok": MODELS is not None, "device": str(MODELS.device) if MODELS else None}


def _decode(data_url: str) -> np.ndarray:
    b64 = data_url.split(",", 1)[1] if "," in data_url else data_url
    buf = np.frombuffer(base64.b64decode(b64), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("bad frame")
    return img


@app.websocket("/ws/{vid}")
async def ws(websocket: WebSocket, vid: str):
    await websocket.accept()
    v = VIDEOS.get(vid)
    if not v:
        await websocket.send_json({"error": "unknown video id; upload again"})
        await websocket.close()
        return
    an = Analyzer(MODELS, v["audio"], v["modality"]["modality"])
    det_conf, person_conf = 0.25, 0.35
    try:
        while True:
            msg = await websocket.receive_json()
            if msg.get("type") == "config":
                if msg.get("modality") in ("rgb", "thermal"):
                    an.set_modality(msg["modality"])
                det_conf = float(msg.get("det_conf", det_conf))
                person_conf = float(msg.get("person_conf", person_conf))
                continue
            try:
                frame = _decode(msg["image"])
            except Exception:
                await websocket.send_json({"skip": True, "seq": msg.get("seq")})
                continue
            t = float(msg.get("t", 0.0))
            res = await run_in_threadpool(an.analyze, frame, t, det_conf, person_conf)
            res["seq"] = msg.get("seq")
            await websocket.send_json(res)
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")


if __name__ == "__main__":
    import argparse
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available)")
    args = ap.parse_args()
    if args.device:
        os.environ["ADAPTFUSE_DEVICE"] = args.device
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
