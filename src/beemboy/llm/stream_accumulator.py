"""Accumulate OpenAI-style chat completion stream into one assistant message dict."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

OnDelta = Callable[[str], None] | None


def _merge_tool_call_deltas(tool_calls_by_index: dict[int, dict[str, Any]], chunk_tool_calls: list[Any]) -> None:
    for p in chunk_tool_calls:
        idx = getattr(p, "index", 0) or 0
        if idx not in tool_calls_by_index:
            tool_calls_by_index[idx] = {"id": "", "name": "", "arguments": ""}
        slot = tool_calls_by_index[idx]
        tid = getattr(p, "id", None)
        if tid:
            slot["id"] = tid
        fn = getattr(p, "function", None)
        if fn is not None:
            name = getattr(fn, "name", None)
            if name:
                slot["name"] = name
            args = getattr(fn, "arguments", None)
            if args:
                slot["arguments"] = slot["arguments"] + args


async def accumulate_chat_stream(
    stream: Any,
    *,
    on_text_delta: OnDelta = None,
) -> dict[str, Any]:
    """Consume an async chat completion stream; return assistant message dict for chat history."""
    content_parts: list[str] = []
    tool_calls_by_index: dict[int, dict[str, Any]] = {}

    async for chunk in stream:
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue
        choice = choices[0]
        delta = getattr(choice, "delta", None)
        if delta is None:
            continue

        piece = getattr(delta, "content", None)
        if piece:
            if on_text_delta:
                on_text_delta(piece)
            content_parts.append(piece)

        tc_list = getattr(delta, "tool_calls", None)
        if tc_list:
            _merge_tool_call_deltas(tool_calls_by_index, tc_list)

    content = "".join(content_parts) if content_parts else None

    if tool_calls_by_index:
        tool_calls: list[dict[str, Any]] = []
        for i in sorted(tool_calls_by_index.keys()):
            slot = tool_calls_by_index[i]
            tool_calls.append(
                {
                    "id": slot["id"] or f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": slot["name"],
                        "arguments": slot["arguments"] or "{}",
                    },
                }
            )
        return {
            "role": "assistant",
            "content": content or "",
            "tool_calls": tool_calls,
        }

    return {"role": "assistant", "content": (content or "").strip()}
