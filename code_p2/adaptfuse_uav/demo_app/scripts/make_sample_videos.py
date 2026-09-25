"""
Build two short test clips from the local training datasets:

    demo_app/samples/sample_rgb.mp4       FLAME/SARD RGB frames + ESC-50 audio
    demo_app/samples/sample_thermal.mp4   the matching pseudo-thermal frames + same audio

Segments: aerial forest without fire (wind audio) -> aerial wildfire (crackling fire)
-> ground people from SARD (crying / distress audio).

Requires the raw datasets under code_p2/adaptfuse_uav/datasets/ (not in git).
Run from code_p2/adaptfuse_uav:
    python demo_app/scripts/make_sample_videos.py
"""

import re
import subprocess
import sys
import wave
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.media import ffmpeg_exe  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "datasets"
OUT = ROOT / "demo_app" / "samples"
FPS, W, H, SR = 10, 640, 360, 16000
ESC = DS / "raw/ESC50/ESC-50-master/audio"


def _num(p: Path) -> int:
    m = re.search(r"(\d+)(?!.*\d)", p.stem)
    return int(m.group(1)) if m else 0


def _cover(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    s = max(W / w, H / h)
    img = cv2.resize(img, (int(w * s + 0.5), int(h * s + 0.5)))
    y, x = (img.shape[0] - H) // 2, (img.shape[1] - W) // 2
    return img[y:y + H, x:x + W]


def _thermal_twin(rgb_path: Path) -> Path:
    rel = rgb_path.relative_to(DS / "raw")
    return (DS / "pseudo_thermal" / rel).with_suffix(".png")


def _esc(cls: int, seconds: float) -> np.ndarray:
    import librosa
    files = sorted(ESC.glob(f"*-{cls}.wav"))
    y = np.concatenate([librosa.load(f, sr=SR, mono=True)[0] for f in files[:3]])
    return y[: int(seconds * SR)]


def segments():
    fire = sorted((DS / "raw/FLAME/Training/fire").glob("resized_frame*.jpg"), key=_num)
    fire = [p for p in fire if 2000 <= _num(p) < 2050]
    nonfire = sorted((DS / "raw/FLAME/Training/non-fire").glob("*.jpg"), key=_num)[:30]
    sard = sorted((DS / "raw/SARD/train/images").glob("*.jpg"))[:4]
    sard = [p for p in sard for _ in range(15)]        # hold each still 1.5 s
    return [(nonfire, 16), (fire, 12), (sard, 20)]      # ESC-50: wind, crackling_fire, crying_baby


def write(kind: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    frames, audio = [], []
    for paths, esc_cls in segments():
        for p in paths:
            src = _thermal_twin(p) if kind == "thermal" else p
            img = cv2.imread(str(src if src.exists() else p))
            if kind == "thermal":
                img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
            frames.append(_cover(img))
        audio.append(_esc(esc_cls, len(paths) / FPS))
    y = np.concatenate(audio)
    wav = OUT / f"_tmp_{kind}.wav"
    with wave.open(str(wav), "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(SR)
        f.writeframes((np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes())
    raw = OUT / f"_tmp_{kind}.mp4"
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for fr in frames:
        vw.write(fr)
    vw.release()
    dst = OUT / f"sample_{kind}.mp4"
    subprocess.run([ffmpeg_exe(), "-v", "error", "-y", "-i", str(raw), "-i", str(wav),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
                    "-movflags", "+faststart", str(dst)], check=True)
    raw.unlink(); wav.unlink()
    print(f"[ok] {dst} ({len(frames) / FPS:.1f} s, {dst.stat().st_size / 1e6:.1f} MB)")
    return dst


if __name__ == "__main__":
    write("rgb")
    write("thermal")
