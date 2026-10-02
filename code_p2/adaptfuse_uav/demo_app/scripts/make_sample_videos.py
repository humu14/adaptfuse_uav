"""
Build the short test clips in demo_app/samples/ from the local datasets:

    sample_rgb.mp4         FLAME forest (wind) -> FLAME wildfire (crackling fire) -> SARD people (crying)
    sample_thermal.mp4     the matching pseudo-thermal frames + the same audio
    sample_collapse.mp4    AIDER collapsed buildings  (rubble crashes, siren, crying)
    sample_flood.mp4       AIDER flooded areas        (rain, thunder, rushing water)
    sample_traffic.mp4     AIDER traffic incidents    (siren, horns, engines)
    sample_city_fire.mp4   AIDER fires                (crackling fire, siren)

The AIDER clips use images from the grouped *test* split only (never seen in training),
picked by size and a fixed seed, not by any model output. Each still gets a slow
pan/zoom and a cross-fade so it plays like aerial footage.

Audio: ESC-50 clips with their silent gaps cut out, cross-faded, loudness-matched and
layered over a quiet rotor bed (helicopter), so every second carries the intended sound.

Requires the raw datasets under code_p2/adaptfuse_uav/datasets/ (not in git).
Run from code_p2/adaptfuse_uav:
    python demo_app/scripts/make_sample_videos.py [names...]
"""

import random
import re
import subprocess
import sys
import wave
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.media import ffmpeg_exe  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "datasets"
OUT = ROOT / "demo_app" / "samples"
TEST_CSV = ROOT / "data/metadata_grouped/seed_42/test.csv"
FPS, W, H, SR = 10, 640, 360, 16000
ESC = DS / "raw/ESC50/ESC-50-master/audio"
# ESC-50 class ids
WIND, RAIN, SEA, FIRE, POUR, THUNDER, CRY, GLASS, HELI, SIREN, HORN, ENGINE, DOOR = \
    16, 10, 11, 12, 17, 19, 20, 39, 40, 42, 43, 44, 30


# ── audio ────────────────────────────────────────────────────────────────────

def _esc_clean(cls: int, top_db: float = 35.0) -> np.ndarray:
    """All ESC-50 clips of one class, silent gaps removed, joined with 40 ms cross-fades."""
    import librosa
    parts = []
    for f in sorted(ESC.glob(f"*-{cls}.wav")):
        y = librosa.load(f, sr=SR, mono=True)[0]
        for a, b in librosa.effects.split(y, top_db=top_db):
            if b - a > 0.15 * SR:
                parts.append(y[a:b])
    out = parts[0]
    for p in parts[1:]:
        out = _xfade(out, p, int(0.04 * SR))
    return out


def _xfade(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    n = min(n, len(a), len(b))
    r = np.linspace(0, 1, n, dtype=np.float32)
    return np.concatenate([a[:-n], a[-n:] * (1 - r) + b[:n] * r, b[n:]])


def _rms_norm(y: np.ndarray, dbfs: float) -> np.ndarray:
    rms = np.sqrt(np.mean(y ** 2)) + 1e-9
    return y * (10 ** (dbfs / 20) / rms)


_CACHE: dict = {}


def layer(cls: int, seconds: float, dbfs: float, offset: float = 0.0) -> np.ndarray:
    """`seconds` of a class's cleaned sound at a target loudness (looped if needed)."""
    if cls not in _CACHE:
        _CACHE[cls] = _esc_clean(cls)
    y = _CACHE[cls]
    n = int(seconds * SR)
    start = int(offset * SR) % len(y)
    reps = int(np.ceil((start + n) / len(y))) + 1
    return _rms_norm(np.tile(y, reps)[start:start + n], dbfs)


def events(cls: int, seconds: float, times: list, dbfs: float, length: float = 1.2) -> np.ndarray:
    """Short bursts of a class at given times (e.g. crashes), silence elsewhere."""
    out = np.zeros(int(seconds * SR), np.float32)
    for k, t in enumerate(times):
        seg = layer(cls, length, dbfs, offset=3.1 * k)
        fade = np.minimum(1, np.minimum(np.arange(len(seg)), np.arange(len(seg))[::-1]) / (0.05 * SR))
        a = int(t * SR)
        b = min(len(out), a + len(seg))
        out[a:b] += (seg * fade)[: b - a]
    return out


def envelope(y: np.ndarray, fade_s: float = 0.4) -> np.ndarray:
    n = min(len(y) // 2, int(fade_s * SR))
    r = np.linspace(0, 1, n, dtype=np.float32)
    y = y.copy()
    y[:n] *= r
    y[-n:] *= r[::-1]
    return y


def delayed(y: np.ndarray, start_s: float, total_s: float) -> np.ndarray:
    out = np.zeros(int(total_s * SR), np.float32)
    a = int(start_s * SR)
    out[a:a + len(y)] = y[: len(out) - a]
    return out


def mix(*tracks) -> np.ndarray:
    n = min(len(t) for t in tracks)
    y = np.sum([t[:n] for t in tracks], axis=0)
    peak = np.abs(y).max()
    return y * (0.89 / peak) if peak > 0.89 else y


# ── video ────────────────────────────────────────────────────────────────────

def _num(p: Path) -> int:
    m = re.search(r"(\d+)(?!.*\d)", p.stem)
    return int(m.group(1)) if m else 0


def _cover(img: np.ndarray, w: int = W, h: int = H) -> np.ndarray:
    ih, iw = img.shape[:2]
    s = max(w / iw, h / ih)
    img = cv2.resize(img, (int(iw * s + 0.5), int(ih * s + 0.5)), interpolation=cv2.INTER_CUBIC)
    y, x = (img.shape[0] - h) // 2, (img.shape[1] - w) // 2
    return img[y:y + h, x:x + w]


def ken_burns(img: np.ndarray, seconds: float, seed: int) -> list:
    """Slow zoom (1.0 -> 1.15) with a small pan, as from a hovering drone."""
    rng = random.Random(seed)
    base = _cover(img, int(W * 1.2), int(H * 1.2))
    n = int(seconds * FPS)
    z0, z1 = (1.0, 1.15) if rng.random() < 0.5 else (1.15, 1.0)
    dx, dy = rng.uniform(-0.06, 0.06), rng.uniform(-0.04, 0.04)
    frames = []
    for i in range(n):
        a = i / max(1, n - 1)
        z = z0 + (z1 - z0) * a
        cw, ch = base.shape[1] / (1.2 * z), base.shape[0] / (1.2 * z)
        cx = base.shape[1] / 2 + dx * base.shape[1] * (a - 0.5)
        cy = base.shape[0] / 2 + dy * base.shape[0] * (a - 0.5)
        M = np.float32([[W / cw, 0, -(cx - cw / 2) * W / cw], [0, H / ch, -(cy - ch / 2) * H / ch]])
        frames.append(cv2.warpAffine(base, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT))
    return frames


def slideshow(images: list, seconds_each: float, xfade_s: float = 0.5, seed: int = 0) -> list:
    out = []
    k = int(xfade_s * FPS)
    for i, img in enumerate(images):
        fr = ken_burns(img, seconds_each + xfade_s, seed + i)
        if out:
            for j in range(k):
                a = (j + 1) / (k + 1)
                out[-k + j] = cv2.addWeighted(out[-k + j], 1 - a, fr[j], a, 0)
            fr = fr[k:]
        out += fr
    return out


def aider_test(cls_folder: str, n: int, seed: int) -> list:
    df = pd.read_csv(TEST_CSV)
    paths = [p for p in df[df.source_dataset == "aider"].rgb_path if Path(p).parent.name == cls_folder]
    sized = [(cv2.imread(p).shape[1], p) for p in paths]
    big = [p for w, p in sorted(sized, reverse=True) if w >= 380]
    random.Random(seed).shuffle(big)
    return [cv2.imread(p) for p in sorted(big[:n])]


# ── clips ────────────────────────────────────────────────────────────────────

def rotor_bed(seconds: float) -> np.ndarray:
    return layer(HELI, seconds, -36)


def clip_collapse():
    imgs = aider_test("collapsed_building", 6, seed=1)
    frames = slideshow(imgs, 3.2)
    d = len(frames) / FPS
    audio = mix(rotor_bed(d), layer(WIND, d, -34),
                events(GLASS, d, [1.0, 4.6, 9.8, 15.2], -20, 1.0),
                events(DOOR, d, [2.2, 7.4, 12.6], -24, 0.8),
                delayed(envelope(layer(SIREN, d * 0.55, -26)), d * 0.45, d),
                events(CRY, d, [6.0, 13.5], -24, 2.5))
    return frames, audio


def clip_flood():
    imgs = aider_test("flooded_areas", 6, seed=2)
    frames = slideshow(imgs, 3.2)
    d = len(frames) / FPS
    audio = mix(rotor_bed(d), layer(RAIN, d, -24), layer(SEA, d, -25, 2.0), layer(POUR, d, -30, 5.0),
                events(THUNDER, d, [3.0, 12.0], -22, 3.0))
    return frames, audio


def clip_traffic():
    imgs = aider_test("traffic_incident", 5, seed=3)
    frames = slideshow(imgs, 3.2)
    d = len(frames) / FPS
    audio = mix(rotor_bed(d), layer(ENGINE, d, -28), envelope(layer(SIREN, d, -23)),
                events(HORN, d, [1.5, 6.5, 11.0], -24, 0.9))
    return frames, audio


def clip_city_fire():
    imgs = aider_test("fire", 5, seed=4)
    frames = slideshow(imgs, 3.2)
    d = len(frames) / FPS
    audio = mix(rotor_bed(d), layer(FIRE, d, -21), envelope(layer(SIREN, d, -29, 4.0)),
                events(GLASS, d, [5.0, 11.5], -25, 0.8))
    return frames, audio


def _thermal_twin(rgb_path: Path) -> Path:
    rel = rgb_path.relative_to(DS / "raw")
    return (DS / "pseudo_thermal" / rel).with_suffix(".png")


def _flame_sard(kind: str):
    fire = sorted((DS / "raw/FLAME/Training/fire").glob("resized_frame*.jpg"), key=_num)
    fire = [p for p in fire if 2000 <= _num(p) < 2050]
    nonfire = sorted((DS / "raw/FLAME/Training/non-fire").glob("*.jpg"), key=_num)[:30]
    sard = sorted((DS / "raw/SARD/train/images").glob("*.jpg"))[:4]
    sard = [p for p in sard for _ in range(15)]          # hold each still 1.5 s
    frames, audio = [], []
    for paths, sound in ((nonfire, lambda d: mix(layer(WIND, d, -24), rotor_bed(d))),
                         (fire, lambda d: mix(layer(FIRE, d, -20), layer(WIND, d, -32), rotor_bed(d))),
                         (sard, lambda d: mix(events(CRY, d, [0.2, 3.2], -20, 2.6), rotor_bed(d)))):
        for p in paths:
            src = _thermal_twin(p) if kind == "thermal" else p
            img = cv2.imread(str(src if src.exists() else p))
            if kind == "thermal":
                img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
            frames.append(_cover(img))
        audio.append(envelope(sound(len(paths) / FPS), 0.15))
    return frames, np.concatenate(audio)


CLIPS = {
    "rgb": lambda: _flame_sard("rgb"),
    "thermal": lambda: _flame_sard("thermal"),
    "collapse": clip_collapse,
    "flood": clip_flood,
    "traffic": clip_traffic,
    "city_fire": clip_city_fire,
}


def write(name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    frames, y = CLIPS[name]()
    y = y[: int(len(frames) / FPS * SR)]
    wav = OUT / f"_tmp_{name}.wav"
    with wave.open(str(wav), "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(SR)
        f.writeframes((np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes())
    raw = OUT / f"_tmp_{name}.mp4"
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for fr in frames:
        vw.write(fr)
    vw.release()
    dst = OUT / f"sample_{name}.mp4"
    subprocess.run([ffmpeg_exe(), "-v", "error", "-y", "-i", str(raw), "-i", str(wav),
                    "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
                    "-shortest", "-movflags", "+faststart", str(dst)], check=True)
    raw.unlink(); wav.unlink()
    print(f"[ok] {dst} ({len(frames) / FPS:.1f} s, {dst.stat().st_size / 1e6:.1f} MB)")
    return dst


if __name__ == "__main__":
    for n in sys.argv[1:] or CLIPS:
        write(n)
