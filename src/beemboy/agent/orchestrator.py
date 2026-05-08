from __future__ import annotations

from collections.abc import Callable
import json
from typing import Any

import structlog

from beemboy.config.settings import Settings
from beemboy.context.compression import ContextCompressor
from beemboy.context.injectors import ClockInjector, LiveContextInjector
from beemboy.llm.llama_server import LlamaServerBackend
from beemboy.memory.store import MemoryStore
from beemboy.mcp.bundle import MCPBundle
from beemboy.prompting.loader import PromptPackLoader

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
        self._memory = MemoryStore(settings.memory_store_path)
        self._compressor = ContextCompressor(enabled=settings.context_compression)
        self._prompt_loader = PromptPackLoader()
        self._recognized_identity: dict[str, Any] | None = None
        self._injectors: list[Any] = [ClockInjector()]
        if settings.live_context_enabled:
            self._injectors.append(LiveContextInjector())

    def set_recognized_identity_context(self, *, person_id: str, name: str, confidence: float) -> None:
        """Optional hook for future camera pipeline integration."""
        if confidence < 0.75:
            self._recognized_identity = None
            return
        self._recognized_identity = {
            "person_id": person_id,
            "name": name,
            "confidence": confidence,
        }
        self._memory.upsert_known_identity(person_id=person_id, name=name)

    def build_system_prompt(self) -> str:
        parts = self._prompt_loader.load_sections()
        if not parts:
            parts = [
                f"You are {self._settings.assistant_name}, a helpful local AI assistant. "
                "Keep replies concise and friendly. Decide first whether a tool is required before calling one. "
                "Only call tools when the user explicitly asks to search/fetch/check time/date, "
                "or when answering confidently requires fresh external data. "
                "If the question is general and can be answered from stable knowledge, do not call tools. "
                "Never claim web/news/time access unless you actually called a tool in this turn.",
            ]
        for inj in self._injectors:
            block = inj.inject(self._settings)
            if block:
                parts.append(block)
        memory_block = self._build_memory_block()
        if memory_block:
            parts.append(memory_block)
        return "\n\n".join(parts)

    def _build_memory_block(self) -> str:
        summary = self._memory.summarize_for_prompt()
        if self._recognized_identity:
            summary += (
                "\nRecognized identity context: "
                f"name={self._recognized_identity['name']}, "
                f"person_id={self._recognized_identity['person_id']}, "
                f"confidence={self._recognized_identity['confidence']:.2f}"
            )
        if self._compressor.should_compress(summary):
            summary = self._compressor.compress_for_context(summary)
        return "Memory context block (internal):\n" + summary

    def _pack_old_history(self, history: list[dict[str, Any]]) -> tuple[dict[str, str] | None, list[dict[str, Any]]]:
        keep_recent_messages = 10
        if len(history) <= keep_recent_messages:
            return None, history

        older = history[:-keep_recent_messages]
        recent = history[-keep_recent_messages:]
        lines: list[str] = []
        for item in older[-24:]:
            role = str(item.get("role") or "unknown")
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            compact = " ".join(content.split())
            if len(compact) > 220:
                compact = compact[:217] + "..."
            lines.append(f"[{role}] {compact}")
        if not lines:
            return None, recent

        packed_body = "\n".join(lines)
        if self._compressor.should_compress(packed_body):
            packed_body = self._compressor.compress_for_context(packed_body)
        packed = "Packed older conversation (internal):\n" + packed_body
        return {"role": "system", "content": packed}, recent

    @staticmethod
    def _is_explicit_tool_command(user_text: str) -> bool:
        text = user_text.strip().lower()
        command_prefixes = (
            "search ",
            "look up ",
            "lookup ",
            "fetch ",
            "find online ",
            "get from web ",
        )
        if text.startswith(command_prefixes):
            return True
        explicit_requests = (
            "what time is it",
            "current time",
            "what day is it",
            "today's date",
            "todays date",
            "check the web",
            "search the web",
            "from the web",
        )
        return any(k in text for k in explicit_requests)

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

        # Prefer explicit web/search requests when available.
        if any(k in text for k in ("search", "web", "look up", "lookup", "find online", "from the web")):
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
        packed_history, recent_history = self._pack_old_history(history)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.build_system_prompt()},
        ]
        if packed_history:
            messages.append(packed_history)
        messages.extend(recent_history)
        messages.append({"role": "user", "content": user_text})
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

            # Safety fallback for explicit command-style tool requests.
            if (
                round_i == 0
                and self._is_explicit_tool_command(user_text)
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
            try:
                self._memory.ingest_turn(user_text, text)
            except Exception:
                log.exception("memory.ingest_failed")
            new_history = [
                item
                for item in messages[1:]
                if not (
                    item.get("role") == "system"
                    and isinstance(item.get("content"), str)
                    and item["content"].startswith("Packed older conversation (internal):")
                )
            ]
            return new_history, text

        return (
            history + [{"role": "user", "content": user_text}],
            "Stopped: max tool rounds exceeded.",
        )
