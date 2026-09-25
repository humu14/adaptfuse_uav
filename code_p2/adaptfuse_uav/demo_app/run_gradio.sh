#!/usr/bin/env bash
# AdapFuse-UAV batch analysis (Gradio) -> http://127.0.0.1:7860
cd "$(dirname "$0")" && exec python gradio_app.py "$@"
