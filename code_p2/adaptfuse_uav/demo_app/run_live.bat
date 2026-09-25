@echo off
REM AdapFuse-UAV live demo -> http://127.0.0.1:8000
cd /d "%~dp0"
python live\server.py %*
