from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from openai.types.chat import ChatCompletion


@runtime_checkable
class ChatBackend(Protocol):
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatCompletion: ...
