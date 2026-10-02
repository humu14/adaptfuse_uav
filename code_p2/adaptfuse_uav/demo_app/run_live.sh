#!/usr/bin/env bash
# AdapFuse-UAV live demo -> http://127.0.0.1:8001
cd "$(dirname "$0")" && exec python live/server.py "$@"
