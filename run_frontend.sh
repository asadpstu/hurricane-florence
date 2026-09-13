#!/usr/bin/env bash
set -euo pipefail

HOST="${FLORENCE_HOST:-127.0.0.1}"
PORT="${FLORENCE_PORT:-8000}"

python -m uvicorn src.frontend.api.main:app \
  --host "$HOST" \
  --port "$PORT" \
  --reload
