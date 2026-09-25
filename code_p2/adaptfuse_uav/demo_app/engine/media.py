"""Video metadata and audio-track extraction via the ffmpeg bundled in imageio-ffmpeg."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


def ffmpeg_exe() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def video_info(path: str | Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"fps": float(fps), "frames": n, "width": w, "height": h,
            "duration": n / fps if fps else 0.0}


def _decode_audio(path: Path, sr: int) -> np.ndarray | None:
    cmd = [ffmpeg_exe(), "-v", "error", "-i", str(path), "-vn", "-ac", "1",
           "-ar", str(sr), "-f", "f32le", "pipe:1"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        return None                                    # no audio stream
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


@dataclass
class AudioTrack:
    """Mono waveform at the two rates the audio networks expect."""

    y16: np.ndarray       # AudioCNN / AdapFuse audio branch (16 kHz)
    y32: np.ndarray       # PANNs Cnn6 (32 kHz)

    def window(self, t: float, seconds: float, sr: int) -> np.ndarray:
        """`seconds` of audio ending at time t (zero-padded at the start)."""
        y = self.y16 if sr == 16000 else self.y32
        n = int(seconds * sr)
        end = min(max(int(round(t * sr)), 0), len(y))
        seg = y[max(0, end - n):end]
        if len(seg) < n:
            seg = np.pad(seg, (n - len(seg), 0))
        return seg


def extract_audio(path: str | Path) -> AudioTrack | None:
    path = Path(path)
    y16 = _decode_audio(path, 16000)
    if y16 is None or len(y16) < 1600:
        return None
    y32 = _decode_audio(path, 32000)
    return AudioTrack(y16=y16, y32=y32)


def transcode_to_h264(src: str | Path, dst: str | Path) -> Path:
    """Re-encode to browser-playable H.264/AAC MP4 (keeps audio if present)."""
    cmd = [ffmpeg_exe(), "-v", "error", "-y", "-i", str(src),
           "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-movflags", "+faststart", str(dst)]
    subprocess.run(cmd, check=True, capture_output=True)
    return Path(dst)
