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

See `.env.example` for `LLAMA_*`, `BRAVE_*`, optional `MCP_SERVERS` JSON, live-context options, and **`STREAM_RESPONSES`** (token streaming to the terminal while the model generates).

By default, if you provide no MCP configuration, Beemboy auto-enables a built-in MCP trio via `uvx`:

- `time` (`mcp-server-time`)
- `fetch` (`mcp-server-fetch`)
- `ddg-search` (`duckduckgo-mcp-server`)

You can disable this fallback with `DEFAULT_MCP_ENABLED=false`. If you want a specific timezone for the time server, set `DEFAULT_TIME_TIMEZONE`.

## MCP in the llama-server Web UI (`mcp-proxy`)

This flow is for the **llama-server built-in Web UI**, which can talk to MCP servers through a small proxy.

**Beemboy** can use the **same** `mcp-proxy` without hand-writing URLs:

- Set **`MCP_PROXY_BASE_URL`** (e.g. `http://127.0.0.1:8001`).
- Set **`MCP_PROXY_CONFIG`** to your `mcp-proxy` JSON (the file with `"mcpServers": { ... }`), *or* set **`MCP_PROXY_SERVERS`** to a comma-separated list of those server names (`time,fetch,ddg-search`).
- Keep **`MCP_PROXY_URL_SUFFIX=mcp`** unless you need **`sse`** endpoints instead.

Alternatively, add each URL manually in **`MCP_SERVERS`**, for example:

`{"id":"time","transport":"http","url":"http://127.0.0.1:8001/servers/time/mcp"}`

### 1. Start `llama-server` with the Web UI MCP hook

Enable the flag that wires the Web UI to MCP (community write-ups refer to this as **`--webui-mcp-proxy`**). Keep **`--jinja`** if you rely on tool-capable chat templates. Example:

```bash
llama-server --jinja --webui-mcp-proxy -m /path/to/model.gguf --host 127.0.0.1 --port 8080
```

### 2. Install `uv`

[`uv`](https://docs.astral.sh/uv/getting-started/installation/) is used to run `uvx` and the proxy.

### 3. MCP server config

Copy [`config/mcp-proxy.example.json`](config/mcp-proxy.example.json) to a directory you control (e.g. `cp config/mcp-proxy.example.json ./mcp-config.json`) and edit it. If you use the **time** server, set **`--local-timezone=...`** to your real timezone.

### 4. Run `mcp-proxy`

From the **same directory** as your config file:

```bash
uvx mcp-proxy --named-server-config mcp-config.json --allow-origin "*" --port 8001 --stateless
```

The proxy prints a URL per named server. For the Web UI, use the endpoint that ends with **`/mcp`**, not **`/sse`**. Example:

- Given: `http://127.0.0.1:8001/servers/time/sse`
- Use in the UI: `http://127.0.0.1:8001/servers/time/mcp`

### 5. Register servers in the Web UI

1. Open the llama-server Web UI.
2. Go to **Settings → MCP → Add New Server**.
3. Add each server URL (the `.../mcp` form), e.g.  
   `http://127.0.0.1:8001/servers/time/mcp`  
   `http://127.0.0.1:8001/servers/fetch/mcp`  
   `http://127.0.0.1:8001/servers/ddg-search/mcp`
4. Save each server and **enable** it with the toggle. Some servers may need **“use llama-server proxy”** enabled if the UI offers it.

After that, those MCP tools should be available in the Web UI chat.

*This section condenses a community-friendly walkthrough; thanks to folks who documented `--webui-mcp-proxy`, the `/sse` → `/mcp` URL tweak, and the llama-server proxy toggle.*
