from __future__ import annotations

import json
import shlex
from functools import lru_cache
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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


MCPServerDefinition = Annotated[StdioMCPServer | HttpMCPServer, Field(discriminator="transport")]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    assistant_name: str = "Beemboy"
    llama_base_url: str = "http://127.0.0.1:8080/v1"
    llama_model: str = "gpt-3.5-turbo"
    llama_api_key: str = "not-needed"
    temperature: float = 0.7
    max_tool_rounds: int = 16
    request_timeout_s: float = 120.0

    brave_api_key: str | None = None
    brave_mcp_enabled: bool = True
    brave_npx_args: str | None = Field(
        default=None,
        description="Override: shell-style args after npx, e.g. '-y @modelcontextprotocol/server-brave-search'",
    )

    mcp_servers: str | None = Field(default=None, validation_alias="MCP_SERVERS")

    live_context_enabled: bool = True
    weather_city: str | None = None
    weather_lat: str | None = None
    weather_lon: str | None = None
    openweathermap_api_key: str | None = None
    weather_units: str = "metric"
    newsapi_key: str | None = None
    news_country: str = "us"
    news_rss_urls: str | None = None

    @field_validator("mcp_servers", mode="before")
    @classmethod
    def _empty_str_none(cls, v: Any) -> Any:
        if v == "":
            return None
        return v

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
                out.append(HttpMCPServer.model_validate(item))
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

    def resolved_mcp_servers(self) -> list[StdioMCPServer | HttpMCPServer]:
        servers: list[StdioMCPServer | HttpMCPServer] = []
        brave = self._brave_stdio_server()
        if brave:
            servers.append(brave)
        servers.extend(self._parse_extra_mcp_servers())
        return servers


@lru_cache
def get_settings() -> Settings:
    return Settings()
