from __future__ import annotations

from collections.abc import Callable
import json
from typing import Any

import structlog

from beemboy.config.settings import Settings
from beemboy.context.injectors import ClockInjector, LiveContextInjector
from beemboy.llm.llama_server import LlamaServerBackend
from beemboy.mcp.bundle import MCPBundle

log = structlog.get_logger(__name__)


def _completion_assistant_as_dict(msg: Any) -> dict[str, Any]:
    d: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
    if msg.tool_calls:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments or "{}",
                },
            }
            for tc in msg.tool_calls
        ]
    return d


class AgentOrchestrator:
    """Facade: system injectors + chat/tool loop against llama-server and MCP."""

    def __init__(
        self,
        settings: Settings,
        llm: LlamaServerBackend,
        mcp: MCPBundle,
    ) -> None:
        self._settings = settings
        self._llm = llm
        self._mcp = mcp
        self._injectors: list[Any] = [ClockInjector()]
        if settings.live_context_enabled:
            self._injectors.append(LiveContextInjector())

    def build_system_prompt(self) -> str:
        parts = [
            f"You are {self._settings.assistant_name}, a helpful local AI assistant. "
            "Keep replies concise and friendly. When web search tools are available, use them "
            "for current events or facts you are unsure about, then summarize. "
            "Never claim web/news/time access unless you actually called a tool in this turn.",
        ]
        for inj in self._injectors:
            block = inj.inject(self._settings)
            if block:
                parts.append(block)
        return "\n\n".join(parts)

    @staticmethod
    def _should_force_tools(user_text: str) -> bool:
        t = user_text.lower()
        triggers = (
            "latest news",
            "news today",
            "today news",
            "current events",
            "breaking news",
            "what time",
            "current time",
            "what day",
            "today date",
            "fetch ",
            "search ",
            "look up",
            "lookup ",
        )
        return any(k in t for k in triggers)

    def _find_tool_name(self, suffix: str) -> str | None:
        for t in self._mcp.tools:
            if t.openai_name.endswith(suffix):
                return t.openai_name
        return None

    @staticmethod
    def _extract_query_from_prompt(user_text: str) -> str:
        text = user_text.strip()
        for p in ("fetch", "search", "look up", "lookup"):
            idx = text.lower().find(p)
            if idx >= 0:
                candidate = text[idx + len(p) :].strip(" ?:")
                if candidate:
                    return candidate
        return text

    async def _heuristic_tool_fallback(
        self,
        user_text: str,
        *,
        on_tool_round_start: Callable[[], None] | None = None,
        on_tool_call: Callable[[str, str], None] | None = None,
    ) -> list[dict[str, Any]] | None:
        """Direct MCP fallback for prompts where the model skipped tool calls."""
        text = user_text.lower()
        tool_messages: list[dict[str, Any]] = []

        # Prefer explicit web/news search tool when available.
        if any(k in text for k in ("latest", "news", "world cup", "search", "web", "fetch", "look up", "lookup")):
            search_tool = self._find_tool_name("__search")
            if search_tool:
                args_obj = {"query": self._extract_query_from_prompt(user_text)}
                args = json.dumps(args_obj, ensure_ascii=False)
                if on_tool_round_start:
                    on_tool_round_start()
                if on_tool_call:
                    on_tool_call(search_tool, args)
                out = await self._mcp.invoke(search_tool, args)
                tool_messages.append({"role": "tool", "tool_call_id": "fallback_search", "content": out})

                # Optional: fetch one top result if model can do it later; here we keep simple/safe.
                return tool_messages

        if any(k in text for k in ("time", "date", "day")):
            time_tool = self._find_tool_name("__get_current_time")
            if time_tool:
                args_obj: dict[str, Any] = {}
                args = json.dumps(args_obj)
                if on_tool_round_start:
                    on_tool_round_start()
                if on_tool_call:
                    on_tool_call(time_tool, args)
                out = await self._mcp.invoke(time_tool, args)
                tool_messages.append({"role": "tool", "tool_call_id": "fallback_time", "content": out})
                return tool_messages

        if "fetch" in text:
            fetch_tool = self._find_tool_name("__fetch")
            if fetch_tool:
                q = self._extract_query_from_prompt(user_text)
                if q.startswith("http://") or q.startswith("https://"):
                    args_obj = {"url": q}
                    args = json.dumps(args_obj)
                    if on_tool_round_start:
                        on_tool_round_start()
                    if on_tool_call:
                        on_tool_call(fetch_tool, args)
                    out = await self._mcp.invoke(fetch_tool, args)
                    tool_messages.append({"role": "tool", "tool_call_id": "fallback_fetch", "content": out})
                    return tool_messages

        return None

    async def run_turn(
        self,
        history: list[dict[str, Any]],
        user_text: str,
        *,
        on_text_delta: Callable[[str], None] | None = None,
        on_tool_round_start: Callable[[], None] | None = None,
        on_tool_call: Callable[[str, str], None] | None = None,
    ) -> tuple[list[dict[str, Any]], str]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.build_system_prompt()},
            *history,
            {"role": "user", "content": user_text},
        ]
        tools_payload = [t.to_openai() for t in self._mcp.tools] if self._mcp.tools else None
        forced_tool_choice: str | dict[str, Any] | None = None
        if tools_payload and self._should_force_tools(user_text):
            forced_tool_choice = "required"

        for round_i in range(self._settings.max_tool_rounds):
            log.debug("agent.llm_round", round=round_i, num_tools=len(self._mcp.tools))
            if self._settings.stream_responses:
                assistant = await self._llm.stream_complete(
                    messages,
                    tools=tools_payload if tools_payload else None,
                    tool_choice=forced_tool_choice if round_i == 0 else None,
                    on_text_delta=on_text_delta,
                )
            else:
                resp = await self._llm.chat(
                    messages,
                    tools=tools_payload if tools_payload else None,
                    tool_choice=forced_tool_choice if round_i == 0 else None,
                )
                assistant = _completion_assistant_as_dict(resp.choices[0].message)

            tool_calls = assistant.get("tool_calls")
            if tool_calls:
                if not self._mcp.tools:
                    log.warning("agent.unexpected_tool_calls", round=round_i)
                    return (
                        history + [{"role": "user", "content": user_text}],
                        "Model requested tools but no MCP servers are connected.",
                    )
                if on_tool_round_start:
                    on_tool_round_start()
                messages.append(assistant)
                for tc in tool_calls:
                    fn = tc.get("function") or {}
                    name = fn.get("name") or ""
                    args = fn.get("arguments") or "{}"
                    if on_tool_call:
                        on_tool_call(name, args)
                    out = await self._mcp.invoke(name, args)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.get("id", ""),
                            "content": out,
                        }
                    )
                continue

            # Safety fallback: if this prompt should use tools but model skipped calls,
            # call a matching MCP tool directly once and continue the loop.
            if (
                round_i == 0
                and forced_tool_choice == "required"
                and tools_payload
                and not tool_calls
            ):
                fallback_msgs = await self._heuristic_tool_fallback(
                    user_text,
                    on_tool_round_start=on_tool_round_start,
                    on_tool_call=on_tool_call,
                )
                if fallback_msgs:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": "Using tool results for a grounded answer.",
                        }
                    )
                    messages.extend(fallback_msgs)
                    continue

            text = (assistant.get("content") or "").strip()
            messages.append({"role": "assistant", "content": text})
            new_history = messages[1:]
            return new_history, text

        return (
            history + [{"role": "user", "content": user_text}],
            "Stopped: max tool rounds exceeded.",
        )
