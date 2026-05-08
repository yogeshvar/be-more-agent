#!/usr/bin/env bash
set -euo pipefail

LLAMA_CLI="${LLAMA_CLI:-$HOME/Mags/llama.cpp/build/bin/llama-cli}"
MODEL="${MODEL:-$HOME/Mags/models/llm/qwen2.5-3b-instruct-q4_k_m.gguf}"

if [[ ! -x "$LLAMA_CLI" ]]; then
  echo "ERROR: $LLAMA_CLI not found or not executable."
  exit 1
fi
if [[ ! -f "$MODEL" ]]; then
  echo "ERROR: $MODEL not found. Run ./setup.sh first."
  exit 1
fi

exec "$LLAMA_CLI" \
  -m "$MODEL" \
  -cnv \
  -t 4 \
  -c 4096 \
  --temp 0.7 \
  -p "You are Mags, a helpful local AI assistant running on a Raspberry Pi. Keep replies short and friendly."
