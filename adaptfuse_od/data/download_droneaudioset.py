"""
DroneAudioSet Downloader & Spectrogram Generator
=================================================
Downloads DroneAudioSet (real UAV audio with human voice, disaster sounds,
and real rotor noise at SNR -57.2 to -2.5 dB) from Hugging Face.

Converts 1D audio waveforms into 2D log-mel spectrograms (128 mel bins × 256 time steps)
ready for AudioYOLO training.

Source: https://huggingface.co/datasets/ahlab-drone-project/DroneAudioSet
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "datasets" / "droneaudioset"


def download_droneaudioset(output_dir: Path):
    """Download DroneAudioSet using Hugging Face datasets."""
    print(f"Downloading DroneAudioSet to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
        dataset = load_dataset("ahlab-drone-project/DroneAudioSet")
        print("✓ Successfully loaded DroneAudioSet from Hugging Face!")
        print(f"  Splits: {list(dataset.keys())}")
        return dataset
    except Exception as e:
        print(f"Hugging Face dataset download error: {e}")
        print("Manual download instructions:")
        print("  1. Visit https://huggingface.co/datasets/ahlab-drone-project/DroneAudioSet")
        print(f"  2. Extract audio files to {output_dir}")
        return None


def audio_to_log_mel_spectrogram(audio_np: np.ndarray, sr: int = 16000,
                                n_mels: int = 128, n_fft: int = 1024,
                                hop_length: int = 256, target_time_steps: int = 256) -> np.ndarray:
    """
    Convert 1D audio array to (128, 256) log-mel spectrogram tensor.
    """
    import librosa

    # Compute Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio_np, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )

    # Convert to dB (log scale)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)

    # Normalize to [0, 1]
    log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-6)

    # Resize/pad to target_time_steps
    if log_mel.shape[1] < target_time_steps:
        pad_width = target_time_steps - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode="constant")
    else:
        log_mel = log_mel[:, :target_time_steps]

    return log_mel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    out_dir = Path(args.output)
    download_droneaudioset(out_dir)

    # Test spectrogram generator on synthetic noise
    dummy_audio = np.random.randn(16000 * 3)  # 3 seconds at 16kHz
    spec = audio_to_log_mel_spectrogram(dummy_audio)
    print(f"Test Spectrogram shape: {spec.shape}")
