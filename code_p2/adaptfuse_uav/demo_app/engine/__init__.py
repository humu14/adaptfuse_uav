"""AdapFuse-UAV demo inference engine (shared by the live and Gradio apps)."""

import os

# Anaconda ships its own libiomp5md.dll next to PyTorch's; without this the
# process aborts with "OMP: Error #15" as soon as both are loaded.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from .models import ModelBundle, load_models
from .media import AudioTrack, extract_audio, video_info
from .modality import detect_modality
from .pipeline import Analyzer

__all__ = [
    "ModelBundle",
    "load_models",
    "AudioTrack",
    "extract_audio",
    "video_info",
    "detect_modality",
    "Analyzer",
]
