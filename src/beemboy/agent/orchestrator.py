from __future__ import annotations

from typing import Any

import structlog

from beemboy.config.settings import Settings
from beemboy.context.injectors import ClockInjector, LiveContextInjector
from beemboy.llm.llama_server import LlamaServerBackend
from beemboy.mcp.bundle import MCPBundle

log = structlog.get_logger(__name__)


def _assistant_message_dict(msg: Any) -> dict[str, Any]:
    d: dict[str, Any] = {"role": "assistant", "content": msg.content}
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
    ) -> tuple[list[dict[str, Any]], str]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.build_system_prompt()},
            *history,
            {"role": "user", "content": user_text},
        ]
        tools_payload = [t.to_openai() for t in self._mcp.tools] if self._mcp.tools else None

        for round_i in range(self._settings.max_tool_rounds):
            log.debug("agent.llm_round", round=round_i, num_tools=len(self._mcp.tools))
            resp = await self._llm.chat(
                messages,
                tools=tools_payload if tools_payload else None,
            )
            msg = resp.choices[0].message

            if msg.tool_calls:
                if not self._mcp.tools:
                    log.warning("agent.unexpected_tool_calls", round=round_i)
                    return (
                        history + [{"role": "user", "content": user_text}],
                        "Model requested tools but no MCP servers are connected.",
                    )
                messages.append(_assistant_message_dict(msg))
                for tc in msg.tool_calls:
                    args = tc.function.arguments or "{}"
                    out = await self._mcp.invoke(tc.function.name, args)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": out,
                        }
                    )
                continue

            text = (msg.content or "").strip()
            messages.append({"role": "assistant", "content": text})
            new_history = messages[1:]
            return new_history, text

        return (
            history + [{"role": "user", "content": user_text}],
            "Stopped: max tool rounds exceeded.",
        )
