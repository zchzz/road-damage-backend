#!/usr/bin/env bash
set -e

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
UVICORN_WORKERS="${UVICORN_WORKERS:-2}"
LOG_LEVEL="${LOG_LEVEL:-info}"

exec uvicorn app.main:app \
  --host "${HOST}" \
  --port "${PORT}" \
  --workers "${UVICORN_WORKERS}" \
  --log-level "${LOG_LEVEL}"