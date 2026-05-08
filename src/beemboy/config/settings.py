from __future__ import annotations

import json
import pathlib
import shlex
from functools import lru_cache
from typing import Annotated, Any, Literal
from urllib.parse import quote

from pydantic import BaseModel, Field, field_validator


class StdioMCPServer(BaseModel):
    model_config = {"extra": "forbid"}

    id: str
    transport: Literal["stdio"] = "stdio"
    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)


class HttpMCPServer(BaseModel):
    model_config = {"extra": "forbid"}

    id: str
    transport: Literal["http"] = "http"
    url: str
    #: ``streamable`` for ``/mcp`` (Streamable HTTP); ``sse`` for legacy ``/sse`` endpoints.
    http_mode: Literal["streamable", "sse"] = "streamable"


MCPServerDefinition = Annotated[StdioMCPServer | HttpMCPServer, Field(discriminator="transport")]


class Settings(BaseModel):
    model_config = {"extra": "ignore"}

    assistant_name: str = "Beemboy"
    llama_base_url: str = "http://127.0.0.1:8080/v1"
    llama_model: str = "gpt-3.5-turbo"
    llama_api_key: str = "not-needed"
    temperature: float = 0.7
    max_tool_rounds: int = 16
    request_timeout_s: float = 120.0
    stream_responses: bool = True

    brave_api_key: str | None = None
    brave_mcp_enabled: bool = True
    brave_npx_args: str | None = Field(
        default=None,
        description="Override: shell-style args after npx, e.g. '-y @modelcontextprotocol/server-brave-search'",
    )

    mcp_servers: str | None = None

    #: Same layout as ``mcp-proxy --named-server-config``: base URL, e.g. ``http://127.0.0.1:8001``
    mcp_proxy_base_url: str | None = None
    #: Path to JSON with top-level ``mcpServers`` object (keys = server names for URL paths).
    mcp_proxy_config_path: str | None = None
    #: Comma-separated server names (same as keys in ``mcpServers``), optional if config path is set.
    mcp_proxy_servers: str | None = None
    #: Path segment after server name: usually ``mcp`` (Streamable HTTP); use ``sse`` if needed.
    mcp_proxy_url_suffix: str = "mcp"
    default_mcp_enabled: bool = True
    default_time_timezone: str | None = None

    live_context_enabled: bool = True
    live_context_ttl_s: int = 300
    weather_city: str | None = None
    weather_lat: str | None = None
    weather_lon: str | None = None
    openweathermap_api_key: str | None = None
    weather_units: str = "metric"
    newsapi_key: str | None = None
    news_country: str = "us"
    news_rss_urls: str | None = None
    memory_store_path: str = ".beemboy_memory.json"
    context_compression: bool = True
    camera_enabled: bool = False
    camera_identity_store_path: str = ".beemboy_identities.json"
    camera_match_threshold: float = 0.82
    camera_unknown_prompt: str = "I don't recognize you yet. What's your name?"
    camera_detector_backend: Literal["opencv", "none"] = "opencv"
    camera_min_face_size_px: int = 80
    voice_sample_rate: int = 16000
    voice_chunk_ms: int = 80
    voice_input_device: str | None = None
    voice_wake_model_path: str = "assets/wakeword/hey_bmo.onnx"
    voice_wake_phrase: str = "hey bmo"
    voice_wake_threshold: float = 0.52
    voice_wake_trigger_level: int = 2
    voice_wake_refractory_s: float = 1.2
    voice_post_wake_timeout_s: float = 6.0
    voice_max_utterance_seconds: float = 8.0
    voice_silence_seconds: float = 1.25
    voice_energy_threshold: float = 0.01
    voice_followup_seconds: float = 5.0
    voice_whisper_binary: str = "whisper-cli"
    voice_whisper_model_path: str = "models/ggml-base.en.bin"
    voice_piper_binary: str = "piper"
    voice_piper_model_path: str = "models/en_US-lessac-medium.onnx"
    voice_playback_command: str | None = None
    ui_assets_path: str = "assets"
    ui_poll_interval_ms: int = 120

    @field_validator(
        "mcp_servers",
        "mcp_proxy_base_url",
        "mcp_proxy_config_path",
        "mcp_proxy_servers",
        mode="before",
    )
    @classmethod
    def _empty_str_none(cls, v: Any) -> Any:
        if v == "":
            return None
        return v

    @field_validator("camera_match_threshold")
    @classmethod
    def _validate_camera_threshold(cls, v: float) -> float:
        if v < 0 or v > 1:
            raise ValueError("CAMERA_MATCH_THRESHOLD must be between 0 and 1")
        return v

    @field_validator("voice_wake_threshold")
    @classmethod
    def _validate_voice_wake_threshold(cls, v: float) -> float:
        if v < 0 or v > 1:
            raise ValueError("VOICE_WAKE_THRESHOLD must be between 0 and 1")
        return v

    @field_validator("ui_poll_interval_ms")
    @classmethod
    def _validate_ui_poll_interval_ms(cls, v: int) -> int:
        if v < 30:
            raise ValueError("UI_POLL_INTERVAL_MS must be >= 30")
        return v

    def _names_from_mcp_proxy_config(self, path: str) -> list[str]:
        p = pathlib.Path(path).expanduser()
        if not p.is_file():
            raise ValueError(f"MCP_PROXY_CONFIG is not a file: {p}")
        with p.open(encoding="utf-8") as f:
            data = json.load(f)
        servers = data.get("mcpServers")
        if servers is None:
            raise ValueError("MCP proxy config must contain an 'mcpServers' object")
        if not isinstance(servers, dict):
            raise ValueError("'mcpServers' must be a JSON object")
        return list(servers.keys())

    def _mcp_proxy_http_servers(self) -> list[HttpMCPServer]:
        base = (self.mcp_proxy_base_url or "").strip().rstrip("/")
        if not base:
            return []

        names: list[str] = []
        cfg_path = (self.mcp_proxy_config_path or "").strip()
        if cfg_path:
            names.extend(self._names_from_mcp_proxy_config(cfg_path))
        extra = (self.mcp_proxy_servers or "").strip()
        if extra:
            names.extend(part.strip() for part in extra.split(",") if part.strip())

        seen: set[str] = set()
        unique: list[str] = []
        for n in names:
            if n in seen:
                continue
            seen.add(n)
            unique.append(n)

        if not unique:
            return []

        suffix = (self.mcp_proxy_url_suffix or "mcp").strip().strip("/")
        http_mode: Literal["streamable", "sse"] = "sse" if suffix == "sse" else "streamable"

        out: list[HttpMCPServer] = []
        for n in unique:
            path_seg = quote(n, safe="")
            url = f"{base}/servers/{path_seg}/{suffix}"
            out.append(HttpMCPServer(id=n, url=url, http_mode=http_mode))
        return out

    def _parse_extra_mcp_servers(self) -> list[StdioMCPServer | HttpMCPServer]:
        if not self.mcp_servers:
            return []
        data = json.loads(self.mcp_servers)
        if not isinstance(data, list):
            raise ValueError("MCP_SERVERS must be a JSON array")
        out: list[StdioMCPServer | HttpMCPServer] = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"MCP_SERVERS[{i}] must be an object")
            t = item.get("transport", "stdio")
            if t == "http":
                it = dict(item)
                if it.get("http_mode") is None and isinstance(it.get("url"), str):
                    u = it["url"].rstrip("/")
                    if u.endswith("/sse"):
                        it["http_mode"] = "sse"
                out.append(HttpMCPServer.model_validate(it))
            else:
                out.append(StdioMCPServer.model_validate(item))
        return out

    def _brave_stdio_server(self) -> StdioMCPServer | None:
        key = (self.brave_api_key or "").strip()
        if not key or not self.brave_mcp_enabled:
            return None
        if self.brave_npx_args:
            args = shlex.split(self.brave_npx_args)
        else:
            args = ["-y", "@modelcontextprotocol/server-brave-search"]
        return StdioMCPServer(
            id="brave",
            command="npx",
            args=args,
            env={"BRAVE_API_KEY": key},
        )

    def _default_stdio_servers(self) -> list[StdioMCPServer]:
        if not self.default_mcp_enabled:
            return []
        time_args = ["mcp-server-time"]
        tz = (self.default_time_timezone or "").strip()
        if tz:
            time_args.append(f"--local-timezone={tz}")
        return [
            StdioMCPServer(id="time", command="uvx", args=time_args),
            StdioMCPServer(id="fetch", command="uvx", args=["mcp-server-fetch"]),
            StdioMCPServer(id="ddg-search", command="uvx", args=["duckduckgo-mcp-server"]),
        ]

    def resolved_mcp_servers(self) -> list[StdioMCPServer | HttpMCPServer]:
        by_id: dict[str, StdioMCPServer | HttpMCPServer] = {}

        def put(s: StdioMCPServer | HttpMCPServer) -> None:
            by_id[s.id] = s

        brave = self._brave_stdio_server()
        if brave:
            put(brave)
        for s in self._mcp_proxy_http_servers():
            put(s)
        for s in self._parse_extra_mcp_servers():
            put(s)
        if not by_id:
            for s in self._default_stdio_servers():
                put(s)
        return list(by_id.values())


@lru_cache
def get_settings() -> Settings:
    settings_path = pathlib.Path("config/settings.json")
    if not settings_path.is_file():
        return Settings()
    with settings_path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("config/settings.json must contain a JSON object")
    return Settings.model_validate(data)
