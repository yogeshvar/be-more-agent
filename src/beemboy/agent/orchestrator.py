from __future__ import annotations

from collections.abc import Callable
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
            "for current events or facts you are unsure about, then summarize.",
        ]
        for inj in self._injectors:
            block = inj.inject(self._settings)
            if block:
                parts.append(block)
        return "\n\n".join(parts)

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

        for round_i in range(self._settings.max_tool_rounds):
            log.debug("agent.llm_round", round=round_i, num_tools=len(self._mcp.tools))
            if self._settings.stream_responses:
                assistant = await self._llm.stream_complete(
                    messages,
                    tools=tools_payload if tools_payload else None,
                    on_text_delta=on_text_delta,
                )
            else:
                resp = await self._llm.chat(
                    messages,
                    tools=tools_payload if tools_payload else None,
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

            text = (assistant.get("content") or "").strip()
            messages.append({"role": "assistant", "content": text})
            new_history = messages[1:]
            return new_history, text

        return (
            history + [{"role": "user", "content": user_text}],
            "Stopped: max tool rounds exceeded.",
        )
