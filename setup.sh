#!/usr/bin/env bash
set -euo pipefail

LLAMA_CLI="${LLAMA_CLI:-$HOME/Mags/llama.cpp/build/bin/llama-cli}"
MODEL_DIR="${MODEL_DIR:-$HOME/Mags/models/llm}"
MODEL_FILE="qwen2.5-3b-instruct-q4_k_m.gguf"
MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/${MODEL_FILE}"

echo "==> Checking llama.cpp build..."
if [[ ! -x "$LLAMA_CLI" ]]; then
  echo "ERROR: $LLAMA_CLI not found or not executable."
  echo "Build llama.cpp under ~/Mags/llama.cpp first, or set LLAMA_CLI to your llama-cli path."
  exit 1
fi

echo "==> Ensuring $MODEL_DIR exists..."
mkdir -p "$MODEL_DIR"

echo "==> Downloading $MODEL_FILE (~2.0 GB, resumable)..."
wget -c -O "$MODEL_DIR/$MODEL_FILE" "$MODEL_URL"

echo "==> Done. Run ./chat.sh to start chatting."
