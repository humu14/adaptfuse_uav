@echo off
REM AdapFuse-UAV live demo -> http://127.0.0.1:8001
cd /d "%~dp0"
python live\server.py %*
