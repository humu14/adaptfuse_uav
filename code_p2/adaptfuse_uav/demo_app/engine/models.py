"""
Load every network the demo needs, once, onto one device.

| Output                         | Network                              | Weights                  |
|--------------------------------|--------------------------------------|--------------------------|
| scene / victim / nuisance, RUE | AdapFuseV1 ensemble (grouped seeds 41+42) | weights/adaptfuse_v1_grouped_s4{1,2}.pth |
| audio-only disaster / victim   | AudioOnlyModel (AudioCNN)            | weights/baseline_audio.pth |
| fire + smoke boxes             | YOLO26s fine-tuned on CLAHE D-Fire   | weights/dfire_yolo26s_clahe.pt |
| person + vehicle boxes         | YOLO26n, COCO-pretrained             | weights/yolo26n_coco.pt  |
| hazard events + zones          | CLIP ViT-B/16 vision tower, zero-shot | weights/hazard_clip/, hazard_text.pt |
| AudioSet sound events          | PANNs Cnn14_16k (or Cnn6), AudioSet-pretrained | weights/panns_cnn14_16k-*.safetensors |
"""

from __future__ import annotations

import csv
import json
import os
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .classifier import SceneClassifier, load_calibration

DEMO_DIR = Path(__file__).resolve().parents[1]      # demo_app/
PROJECT_DIR = DEMO_DIR.parent                        # code_p2/adaptfuse_uav/
WEIGHTS_DIR = DEMO_DIR / "weights"
DETECTOR_FILE = "dfire_yolo26s_clahe.pt"          # YOLO26s fine-tuned on CLAHE D-Fire (mAP@50 0.771)
DETECTOR_NAMES = {0: "smoke", 1: "fire"}           # official D-Fire order
PERSON_FILE = "yolo26n_coco.pt"
DETECTOR_IMGSZ = 640
CLASSIFIER_FILES = ("adaptfuse_v1_grouped_s41.pth", "adaptfuse_v1_grouped_s42.pth")
CALIBRATION_FILE = "calibration.json"
HAZARD_CAL_FILE = "hazard_calibration.json"
SOUND_FILE = "sound_thresholds.json"
DEFAULT_TAGGER, DEFAULT_TAG_WINDOW = "cnn14_16k", 3.0


def check_detector_names(names: dict, expected: dict, path, exact: bool) -> None:
    """Refuse detectors whose label map differs from the dataset's (e.g. swapped fire/smoke)."""
    got = {int(k): str(v) for k, v in dict(names).items()}
    bad = any(got.get(k) != v for k, v in expected.items()) or (exact and len(got) != len(expected))
    if bad:
        raise ValueError(
            f"{Path(path).name}: class names {got} do not match expected {expected}. "
            "Refusing to load a detector whose labels may be swapped."
        )

# The research code imports itself as `models.*`, so its root must be on sys.path.
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


# ─────────────────────────────────────────────────────────────────────────────
# PANNs Cnn6 (Kong et al., 2020) — exact layer layout of the released checkpoint
# ─────────────────────────────────────────────────────────────────────────────

class _ConvBlock5x5(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 5, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.avg_pool2d(F.relu(self.bn1(self.conv1(x))), 2)


class _ConvBlock3x3(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x, pool=(2, 2)):
        x = F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))))
        return F.avg_pool2d(x, pool) if pool != (1, 1) else x


class _STFT(nn.Module):
    """torchlibrosa-compatible STFT: conv1d with the checkpoint's DFT kernels."""

    def __init__(self, n_fft: int = 1024, hop: int = 320):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        self.conv_real = nn.Conv1d(1, n_fft // 2 + 1, n_fft, stride=hop, bias=False)
        self.conv_imag = nn.Conv1d(1, n_fft // 2 + 1, n_fft, stride=hop, bias=False)

    def forward(self, wav):                              # (B, L)
        x = F.pad(wav[:, None, :], (self.n_fft // 2, self.n_fft // 2), mode="reflect")
        return self.conv_real(x) ** 2 + self.conv_imag(x) ** 2   # power (B, F, T)


class _SpecExtractor(nn.Module):
    def __init__(self, n_fft: int = 1024, hop: int = 320):
        super().__init__()
        self.stft = _STFT(n_fft, hop)


class _LogMel(nn.Module):
    def __init__(self, n_fft: int = 1024):
        super().__init__()
        self.melW = nn.Parameter(torch.zeros(n_fft // 2 + 1, 64), requires_grad=False)

    def forward(self, power):                            # (B, F, T)
        mel = torch.matmul(power.transpose(1, 2), self.melW)   # (B, T, 64)
        return 10.0 * torch.log10(torch.clamp(mel, min=1e-10))


class PANNsCnn6Tagger(nn.Module):
    """32 kHz waveform -> 527 AudioSet clip-level probabilities."""

    SAMPLE_RATE = 32000

    def __init__(self):
        super().__init__()
        self.spectrogram_extractor = _SpecExtractor()
        self.logmel_extractor = _LogMel()
        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = _ConvBlock5x5(1, 64)
        self.conv_block2 = _ConvBlock5x5(64, 128)
        self.conv_block3 = _ConvBlock5x5(128, 256)
        self.conv_block4 = _ConvBlock5x5(256, 512)
        self.fc1 = nn.Linear(512, 512)
        self.fc_audioset = nn.Linear(512, 527)

    def forward(self, wav):                              # (B, L)
        x = self.logmel_extractor(self.spectrogram_extractor.stft(wav))  # (B, T, 64)
        x = x[:, None]                                   # (B, 1, T, 64)
        x = self.bn0(x.transpose(1, 3)).transpose(1, 3)
        for block in (self.conv_block1, self.conv_block2, self.conv_block3, self.conv_block4):
            x = block(x)
        x = x.mean(dim=3)                                # (B, 512, T')
        x = x.max(dim=2).values + x.mean(dim=2)
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc_audioset(x))


class PANNsCnn14_16k(nn.Module):
    """16 kHz waveform -> 527 AudioSet clip-level probabilities (Cnn14_16k, mAP 0.438)."""

    SAMPLE_RATE = 16000

    def __init__(self):
        super().__init__()
        self.spectrogram_extractor = _SpecExtractor(n_fft=512, hop=160)
        self.logmel_extractor = _LogMel(n_fft=512)
        self.bn0 = nn.BatchNorm2d(64)
        chans = (1, 64, 128, 256, 512, 1024, 2048)
        for i in range(6):
            setattr(self, f"conv_block{i + 1}", _ConvBlock3x3(chans[i], chans[i + 1]))
        self.fc1 = nn.Linear(2048, 2048)
        self.fc_audioset = nn.Linear(2048, 527)

    def forward(self, wav):                              # (B, L)
        x = self.logmel_extractor(self.spectrogram_extractor.stft(wav))[:, None]   # (B, 1, T, 64)
        x = self.bn0(x.transpose(1, 3)).transpose(1, 3)
        for i in range(1, 7):
            x = getattr(self, f"conv_block{i}")(x, pool=(2, 2) if i < 6 else (1, 1))
        x = x.mean(dim=3)
        x = x.max(dim=2).values + x.mean(dim=2)
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc_audioset(x))


# ─────────────────────────────────────────────────────────────────────────────
# Bundle
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelBundle:
    device: torch.device
    classifier: object          # SceneClassifier
    audio_only: nn.Module
    fire_det: object          # ultralytics.YOLO
    person_det: object        # ultralytics.YOLO (COCO: person + vehicles)
    tagger: nn.Module
    audioset_names: list
    hazard: object = None     # hazard.HazardModel
    sound_thr: dict = field(default_factory=dict)
    tag_window: float = DEFAULT_TAG_WINDOW
    # One GPU, several websocket sessions: serialize forward passes.
    lock: threading.Lock = field(default_factory=threading.Lock)


def save_shards(state: dict, out_dir: Path, prefix: str, max_bytes: int = 85_000_000) -> list:
    """Split a state dict into <prefix>-0000N.safetensors files below max_bytes each."""
    from safetensors.torch import save_file
    shards, cur, size = [], {}, 0
    for k, v in state.items():
        n = v.numel() * v.element_size()
        if cur and size + n > max_bytes:
            shards.append(cur); cur, size = {}, 0
        cur[k], size = v.contiguous(), size + n
    shards.append(cur)
    for old in Path(out_dir).glob(f"{prefix}-*.safetensors"):
        old.unlink()
    files = []
    for i, sh in enumerate(shards):
        f = Path(out_dir) / f"{prefix}-{i + 1:05d}.safetensors"
        save_file(sh, str(f))
        files.append(f)
    return files


def load_shards(out_dir: Path, prefix: str) -> dict:
    from safetensors.torch import load_file
    files = sorted(Path(out_dir).glob(f"{prefix}-*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no {prefix}-*.safetensors in {out_dir}; run scripts/export_audio.py")
    sd = {}
    for f in files:
        sd.update(load_file(str(f)))
    return sd


def load_tagger(kind: str) -> nn.Module:
    """AudioSet tagger by name: 'cnn14_16k' (mAP 0.438, 16 kHz) or 'cnn6' (mAP 0.343, 32 kHz)."""
    if kind == "cnn14_16k":
        m = PANNsCnn14_16k()
        m.load_state_dict({k: v.float() for k, v in load_shards(WEIGHTS_DIR, "panns_cnn14_16k").items()})
    elif kind == "cnn6":
        m = PANNsCnn6Tagger()
        m.load_state_dict(torch.load(_require(WEIGHTS_DIR / "panns_cnn6.pth"), map_location="cpu",
                                     weights_only=False)["model"])
    else:
        raise ValueError(f"unknown tagger {kind!r}")
    return m.eval()


def load_sound_config(path: Path) -> dict:
    default = {"tagger": DEFAULT_TAGGER, "window_s": DEFAULT_TAG_WINDOW, "thresholds": {}}
    path = Path(path)
    if not path.exists():
        print(f"[models] WARNING: {path.name} not found - {DEFAULT_TAGGER}, default sound thresholds")
        return default
    return {**default, **json.loads(path.read_text(encoding="utf-8"))}


def _load_state(model: nn.Module, path: Path) -> nn.Module:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state"])
    return model


def resolve_members(files, weights_dir: Path) -> list:
    found = []
    for name in files:
        p = Path(weights_dir) / name
        if p.exists():
            found.append(p)
        else:
            print(f"[models] WARNING: classifier member {name} missing - ensemble runs without it")
    if not found:
        raise FileNotFoundError(f"No classifier member found in {weights_dir} (expected {list(files)})")
    return found


def load_member(path: Path, device) -> nn.Module:
    from models.full_model import AdapFuseV1
    m = AdapFuseV1(num_disaster_classes=4, num_victim_classes=2, pretrained=False)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in ck["model_state"].items()})
    return m.to(device).eval()


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing weight file {path}. The demo weights ship with the repository "
            f"under demo_app/weights/; re-clone or run demo_app/scripts/export_weights.py."
        )
    return path


def load_models(device: str | None = None) -> ModelBundle:
    from ultralytics import YOLO
    from models.backbones.audio_backbone import AudioOnlyModel

    device = device or os.environ.get("ADAPTFUSE_DEVICE")
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Members are built with pretrained=False: ImageNet init is irrelevant once the trained
    # state is loaded, and it avoids a torchvision download on a fresh clone.
    cal = load_calibration(WEIGHTS_DIR / CALIBRATION_FILE)
    member_paths = resolve_members(cal["members"] or CLASSIFIER_FILES, WEIGHTS_DIR)
    classifier = SceneClassifier([load_member(p, dev) for p in member_paths], dev,
                                 tta=bool(cal["tta"]), bias=cal["bias"])
    print(f"[models] scene classifier: {[p.name for p in member_paths]}, tta={cal['tta']}, "
          f"bias={'yes' if cal['bias'] else 'no'} (stage {cal['stage']})")

    audio_only = AudioOnlyModel(num_disaster_classes=4, num_victim_classes=2, backbone="audiocnn")
    _load_state(audio_only, _require(WEIGHTS_DIR / "baseline_audio.pth"))

    snd = load_sound_config(WEIGHTS_DIR / SOUND_FILE)
    tagger = load_tagger(snd["tagger"])
    print(f"[models] sound tagger: {snd['tagger']}, {snd['window_s']:.0f} s window")

    from .hazard import HazardModel, load_hazard_calibration
    hazard = HazardModel(_require(WEIGHTS_DIR / "hazard_clip"), _require(WEIGHTS_DIR / "hazard_text.pt"), dev,
                         load_hazard_calibration(WEIGHTS_DIR / HAZARD_CAL_FILE))

    fire_path = _require(WEIGHTS_DIR / DETECTOR_FILE)
    fire_det = YOLO(str(fire_path))
    check_detector_names(fire_det.names, DETECTOR_NAMES, fire_path, exact=True)
    person_path = _require(WEIGHTS_DIR / PERSON_FILE)
    person_det = YOLO(str(person_path))
    check_detector_names(person_det.names, {0: "person", 2: "car", 8: "boat"}, person_path, exact=False)

    with open(_require(WEIGHTS_DIR / "audioset_labels.csv"), newline="", encoding="utf-8") as f:
        names = [row["display_name"] for row in csv.DictReader(f)]

    for m in (audio_only, tagger):
        m.to(dev).eval()

    return ModelBundle(dev, classifier, audio_only, fire_det, person_det, tagger, names,
                       hazard=hazard, sound_thr=snd["thresholds"], tag_window=float(snd["window_s"]))
