@echo off
REM AdapFuse-UAV batch analysis (Gradio) -> http://127.0.0.1:7860
cd /d "%~dp0"
python gradio_app.py %*
