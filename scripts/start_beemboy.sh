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
WAKEWORD_DIR="${WAKEWORD_DIR:-$REPO_ROOT/assets/wakeword}"
WAKEWORD_MODEL_PATH="${WAKEWORD_MODEL_PATH:-$WAKEWORD_DIR/hey_bmo.onnx}"
WHISPER_MODEL_PATH="${WHISPER_MODEL_PATH:-$VOICE_MODELS_DIR/ggml-base.en.bin}"
PIPER_MODEL_PATH="${PIPER_MODEL_PATH:-$VOICE_MODELS_DIR/en_US-lessac-medium.onnx}"
PIPER_MODEL_CONFIG_PATH="${PIPER_MODEL_CONFIG_PATH:-${PIPER_MODEL_PATH}.json}"
WAKEWORD_MODEL_URL="${WAKEWORD_MODEL_URL:-https://raw.githubusercontent.com/brenpoly/be-more-agent/main/wakeword.onnx}"
WHISPER_MODEL_URL="${WHISPER_MODEL_URL:-https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin}"
PIPER_MODEL_URL="${PIPER_MODEL_URL:-https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx}"
PIPER_MODEL_CONFIG_URL="${PIPER_MODEL_CONFIG_URL:-https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json}"
WHISPER_CPP_DIR="${WHISPER_CPP_DIR:-/home/mags/Mags/whisper.cpp}"
WHISPER_BIN_PATH="${WHISPER_BIN_PATH:-$LLAMA_BIN_DIR/whisper-cli}"
PIPER_BIN_PATH="${PIPER_BIN_PATH:-$VENV_DIR/bin/piper}"
SETTINGS_FILE="${SETTINGS_FILE:-$REPO_ROOT/config/settings.json}"

set_settings_kv() {
  local key="$1"
  local value="$2"
  python3 - "$SETTINGS_FILE" "$key" "$value" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
key = sys.argv[2]
value = sys.argv[3]
path.parent.mkdir(parents=True, exist_ok=True)
data = {}
if path.exists():
    with path.open(encoding="utf-8") as f:
        loaded = json.load(f)
    if isinstance(loaded, dict):
        data = loaded
data[key] = value
with path.open("w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
    f.write("\n")
PY
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
mkdir -p "$WAKEWORD_DIR"
if [[ ! -f "$WAKEWORD_MODEL_PATH" ]]; then
  echo "==> Downloading wakeword model to $WAKEWORD_MODEL_PATH"
  curl -L --fail --retry 3 -o "$WAKEWORD_MODEL_PATH" "$WAKEWORD_MODEL_URL"
else
  echo "==> Wakeword model already present: $WAKEWORD_MODEL_PATH"
fi
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
if [[ ! -f "$PIPER_MODEL_CONFIG_PATH" ]]; then
  echo "==> Downloading piper model config to $PIPER_MODEL_CONFIG_PATH"
  curl -L --fail --retry 3 -o "$PIPER_MODEL_CONFIG_PATH" "$PIPER_MODEL_CONFIG_URL"
else
  echo "==> Piper model config already present: $PIPER_MODEL_CONFIG_PATH"
fi

if command -v whisper-cli >/dev/null 2>&1; then
  WHISPER_BIN_PATH="$(command -v whisper-cli)"
  echo "==> Found whisper-cli on PATH: $WHISPER_BIN_PATH"
elif [[ -x "$WHISPER_BIN_PATH" ]]; then
  echo "==> Found whisper-cli at: $WHISPER_BIN_PATH"
else
  echo "==> whisper-cli not found; bootstrapping whisper.cpp at $WHISPER_CPP_DIR"
  if ! command -v git >/dev/null 2>&1 || ! command -v cmake >/dev/null 2>&1 || ! command -v cc >/dev/null 2>&1; then
    if command -v apt-get >/dev/null 2>&1; then
      echo "==> Installing whisper.cpp build prerequisites (git, cmake, build-essential)"
      sudo apt-get update
      sudo apt-get install -y git cmake build-essential
    else
      echo "ERROR: Missing build prerequisites (git/cmake/compiler) and apt-get is unavailable."
      echo "Install git, cmake, and build tools manually, then rerun."
      exit 1
    fi
  fi
  if [[ ! -d "$WHISPER_CPP_DIR/.git" ]]; then
    git clone --depth 1 https://github.com/ggml-org/whisper.cpp "$WHISPER_CPP_DIR"
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

if command -v piper >/dev/null 2>&1; then
  PIPER_BIN_PATH="$(command -v piper)"
  echo "==> Found piper on PATH: $PIPER_BIN_PATH"
elif [[ -x "$PIPER_BIN_PATH" ]]; then
  echo "==> Found piper at: $PIPER_BIN_PATH"
else
  echo "==> piper not found; installing piper-tts into virtualenv"
  "$VENV_DIR/bin/pip" install --upgrade piper-tts
  if [[ -x "$VENV_DIR/bin/piper" ]]; then
    PIPER_BIN_PATH="$VENV_DIR/bin/piper"
  else
    echo "ERROR: piper install succeeded but binary was not found at $VENV_DIR/bin/piper"
    exit 1
  fi
fi

PLAYBACK_CMD=""
if command -v pw-play >/dev/null 2>&1; then
  PLAYBACK_CMD="pw-play"
elif command -v paplay >/dev/null 2>&1; then
  PLAYBACK_CMD="paplay"
elif command -v aplay >/dev/null 2>&1; then
  PLAYBACK_CMD="aplay"
fi

echo "==> Configuring voice settings in $SETTINGS_FILE"
set_settings_kv "voice_wake_model_path" "$WAKEWORD_MODEL_PATH"
set_settings_kv "voice_wake_phrase" "hey bmo"
set_settings_kv "voice_whisper_binary" "$WHISPER_BIN_PATH"
set_settings_kv "voice_piper_binary" "$PIPER_BIN_PATH"
set_settings_kv "voice_whisper_model_path" "$WHISPER_MODEL_PATH"
set_settings_kv "voice_piper_model_path" "$PIPER_MODEL_PATH"
if [[ -n "$PLAYBACK_CMD" ]]; then
  set_settings_kv "voice_playback_command" "$PLAYBACK_CMD"
fi

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
