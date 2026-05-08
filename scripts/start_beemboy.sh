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
VOICE_MODELS_DIR="${VOICE_MODELS_DIR:-$REPO_ROOT/models}"
WHISPER_MODEL_PATH="${WHISPER_MODEL_PATH:-$VOICE_MODELS_DIR/ggml-base.en.bin}"
PIPER_MODEL_PATH="${PIPER_MODEL_PATH:-$VOICE_MODELS_DIR/en_US-lessac-medium.onnx}"
WHISPER_MODEL_URL="${WHISPER_MODEL_URL:-https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin}"
PIPER_MODEL_URL="${PIPER_MODEL_URL:-https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx}"
WHISPER_CPP_DIR="${WHISPER_CPP_DIR:-/home/mags/Mags/whisper.cpp}"
WHISPER_BIN_PATH="${WHISPER_BIN_PATH:-$LLAMA_BIN_DIR/whisper-cli}"
ENV_FILE="${ENV_FILE:-$REPO_ROOT/.env}"

set_env_kv() {
  local key="$1"
  local value="$2"
  if [[ ! -f "$ENV_FILE" ]]; then
    touch "$ENV_FILE"
  fi
  if grep -qE "^${key}=" "$ENV_FILE"; then
    sed -i "s|^${key}=.*|${key}=${value}|" "$ENV_FILE"
  else
    echo "${key}=${value}" >> "$ENV_FILE"
  fi
}

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

echo "==> Ensuring voice model directory at $VOICE_MODELS_DIR"
mkdir -p "$VOICE_MODELS_DIR"
if [[ ! -f "$WHISPER_MODEL_PATH" ]]; then
  echo "==> Downloading whisper model to $WHISPER_MODEL_PATH"
  curl -L --fail --retry 3 -o "$WHISPER_MODEL_PATH" "$WHISPER_MODEL_URL"
else
  echo "==> Whisper model already present: $WHISPER_MODEL_PATH"
fi
if [[ ! -f "$PIPER_MODEL_PATH" ]]; then
  echo "==> Downloading piper model to $PIPER_MODEL_PATH"
  curl -L --fail --retry 3 -o "$PIPER_MODEL_PATH" "$PIPER_MODEL_URL"
else
  echo "==> Piper model already present: $PIPER_MODEL_PATH"
fi

if command -v whisper-cli >/dev/null 2>&1; then
  WHISPER_BIN_PATH="$(command -v whisper-cli)"
  echo "==> Found whisper-cli on PATH: $WHISPER_BIN_PATH"
elif [[ -x "$WHISPER_BIN_PATH" ]]; then
  echo "==> Found whisper-cli at: $WHISPER_BIN_PATH"
else
  echo "==> whisper-cli not found; bootstrapping whisper.cpp at $WHISPER_CPP_DIR"
  if [[ ! -d "$WHISPER_CPP_DIR/.git" ]]; then
    git clone --depth 1 https://github.com/ggml-org/whisper.cpp "$WHISPER_CPP_DIR"
  fi
  if ! command -v cmake >/dev/null 2>&1; then
    echo "ERROR: cmake is required to build whisper-cli. Install cmake and rerun."
    exit 1
  fi
  cmake -S "$WHISPER_CPP_DIR" -B "$WHISPER_CPP_DIR/build"
  cmake --build "$WHISPER_CPP_DIR/build" --target whisper-cli -j
  if [[ -x "$WHISPER_CPP_DIR/build/bin/whisper-cli" ]]; then
    WHISPER_BIN_PATH="$WHISPER_CPP_DIR/build/bin/whisper-cli"
  elif [[ -x "$WHISPER_CPP_DIR/build/whisper-cli" ]]; then
    WHISPER_BIN_PATH="$WHISPER_CPP_DIR/build/whisper-cli"
  else
    echo "ERROR: whisper-cli build completed but binary was not found."
    exit 1
  fi
fi

echo "==> Configuring voice binaries in $ENV_FILE"
set_env_kv "VOICE_WHISPER_BINARY" "$WHISPER_BIN_PATH"
set_env_kv "VOICE_WHISPER_MODEL_PATH" "$WHISPER_MODEL_PATH"
set_env_kv "VOICE_PIPER_MODEL_PATH" "$PIPER_MODEL_PATH"

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
