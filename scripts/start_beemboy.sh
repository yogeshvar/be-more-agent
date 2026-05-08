#!/usr/bin/env bash
set -euo pipefail

# Adjust if your paths differ.
LLAMA_BIN_DIR="${LLAMA_BIN_DIR:-/home/mags/Mags/llama.cpp/build/bin}"
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-$LLAMA_BIN_DIR/llama-server}"
MODEL_PATH="${MODEL_PATH:-/home/mags/Mags/models/llm/llama-3.2-1b-instruct-q4_k_m.gguf}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
LLAMA_LOG="${LLAMA_LOG:-/tmp/llama-server.log}"

# Repository root (one level above this script directory).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"

if [[ ! -x "$LLAMA_SERVER_BIN" ]]; then
  echo "ERROR: llama-server not executable at: $LLAMA_SERVER_BIN"
  exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "ERROR: model not found: $MODEL_PATH"
  exit 1
fi

echo "==> Ensuring Python virtualenv at $VENV_DIR"
if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

echo "==> Upgrading pip and reinstalling Beemboy editable package"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/pip" install -e "$REPO_ROOT"

echo "==> Clearing port $PORT if occupied"
pids="$(lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)"
if [[ -n "$pids" ]]; then
  echo "==> Killing existing listener(s) on port $PORT: $pids"
  # shellcheck disable=SC2086
  kill $pids || true
  sleep 1
  still_up="$(lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -n "$still_up" ]]; then
    echo "==> Forcing kill on stubborn process(es): $still_up"
    # shellcheck disable=SC2086
    kill -9 $still_up || true
  fi
fi

echo "==> Starting llama-server in background"
nohup "$LLAMA_SERVER_BIN" \
  --jinja \
  -m "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  > "$LLAMA_LOG" 2>&1 &
echo "    pid: $!"

echo "==> Tail logs with: tail -f $LLAMA_LOG"
