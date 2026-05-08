# Beemboy

Python assistant that talks to **llama.cpp `llama-server`** (OpenAI-compatible HTTP API) and runs an agent loop with **MCP** tools (default: **Brave Search** when `BRAVE_API_KEY` is set).

## Prerequisites

- `llama-server` built from [llama.cpp](https://github.com/ggml-org/llama.cpp), started with **`--jinja`** and a tool-capable chat template.
- Python 3.11+
- For Brave MCP: Node + `npx` on `PATH`

## Setup

```bash
cd /path/to/be-more-agent
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Copy `.env.example` to `.env` and adjust.

## Run

1. Start the server (example):

   `llama-server --jinja -m /path/to/model.gguf --host 127.0.0.1 --port 8080`

2. Chat:

   `beemboy` (or `python -m beemboy`)

## Configuration

See `.env.example` for `LLAMA_*`, `BRAVE_*`, optional `MCP_SERVERS` JSON, and live-context options.
