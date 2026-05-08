from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from openai.types.chat import ChatCompletion


@runtime_checkable
class ChatBackend(Protocol):
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatCompletion: ...

    async def stream_complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        on_text_delta: Callable[[str], None] | None = None,
    ) -> dict[str, Any]: ...
