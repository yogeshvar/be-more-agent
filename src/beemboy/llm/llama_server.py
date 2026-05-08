from __future__ import annotations

from typing import Any

import structlog
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from beemboy.config.settings import Settings

log = structlog.get_logger(__name__)


class LlamaServerBackend:
    """OpenAI-compatible HTTP client for llama.cpp `llama-server`."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = AsyncOpenAI(
            base_url=settings.llama_base_url.rstrip("/"),
            api_key=settings.llama_api_key,
            timeout=settings.request_timeout_s,
        )
        self._model = settings.llama_model
        self._temperature = settings.temperature

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatCompletion:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
        }
        if tools:
            kwargs["tools"] = tools
        log.debug("llm.request", tool_count=len(tools or []))
        return await self._client.chat.completions.create(**kwargs)
