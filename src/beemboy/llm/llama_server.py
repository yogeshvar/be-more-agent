from __future__ import annotations

from collections.abc import Callable
from typing import Any

import structlog
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from beemboy.config.settings import Settings
from beemboy.llm.stream_accumulator import accumulate_chat_stream

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
        tool_choice: str | dict[str, Any] | None = None,
    ) -> ChatCompletion:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
        }
        if tools:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        log.debug("llm.request", tool_count=len(tools or []))
        return await self._client.chat.completions.create(**kwargs)

    async def stream_complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        *,
        on_text_delta: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """Stream one completion; return assistant message dict (``content`` / ``tool_calls``)."""
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        log.debug("llm.request_stream", tool_count=len(tools or []))
        stream = await self._client.chat.completions.create(**kwargs)
        return await accumulate_chat_stream(stream, on_text_delta=on_text_delta)
