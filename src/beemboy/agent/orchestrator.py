from __future__ import annotations

from collections.abc import Callable
import json
from time import perf_counter
from typing import Any

import structlog

from beemboy.agent.telemetry import TurnTelemetry, estimate_message_chars, estimate_message_tokens, estimate_tokens_from_text
from beemboy.config.settings import Settings
from beemboy.context.compression import ContextCompressor
from beemboy.context.injectors import ClockInjector, LiveContextInjector
from beemboy.llm.llama_server import LlamaServerBackend
from beemboy.memory.store import MemoryStore
from beemboy.mcp.bundle import MCPBundle
from beemboy.prompting.loader import PromptPackLoader
from beemboy.vision.detector import FaceDetector
from beemboy.vision.pipeline import CameraIdentityPipeline
from beemboy.vision.registry import FaceRegistry

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
        self._camera_identity: CameraIdentityPipeline | None = None
        self._injectors: list[Any] = [ClockInjector()]
        if settings.live_context_enabled:
            self._injectors.append(LiveContextInjector())
        if settings.camera_enabled:
            self._camera_identity = CameraIdentityPipeline(
                registry=FaceRegistry(settings.camera_identity_store_path),
                detector=FaceDetector(
                    backend=settings.camera_detector_backend,
                    min_face_size_px=settings.camera_min_face_size_px,
                ),
                match_threshold=settings.camera_match_threshold,
            )

    @property
    def camera_enabled(self) -> bool:
        return self._camera_identity is not None

    def save_state(self) -> None:
        self._memory.save()
        if self._camera_identity:
            self._camera_identity.save_registry()

    def list_known_identities(self) -> list[dict[str, str]]:
        by_person_id: dict[str, dict[str, str]] = {}
        for item in self._memory.state.known_identities:
            if not item.person_id or not item.name:
                continue
            by_person_id[item.person_id] = {
                "person_id": item.person_id,
                "name": item.name,
                "source": "memory",
                "last_seen_at": item.last_seen_at,
            }
        if self._camera_identity:
            for rec in self._camera_identity.list_identities():
                by_person_id[rec.person_id] = {
                    "person_id": rec.person_id,
                    "name": rec.name,
                    "source": "registry",
                    "last_seen_at": rec.last_seen_at,
                }
        identities = list(by_person_id.values())
        identities.sort(key=lambda row: row.get("last_seen_at", ""), reverse=True)
        return identities

    def shutdown(self) -> None:
        self.save_state()

    def set_recognized_identity_context(
        self,
        *,
        person_id: str,
        name: str,
        confidence: float,
        face_embeddings_ref: str | None = None,
    ) -> None:
        """Current known-person context for greeting personalization."""
        if confidence < 0.75:
            self._recognized_identity = None
            return
        self._recognized_identity = {
            "person_id": person_id,
            "name": name,
            "confidence": confidence,
        }
        self._memory.upsert_known_identity(
            person_id=person_id,
            name=name,
            face_embeddings_ref=face_embeddings_ref,
        )

    def observe_camera_embedding(self, embedding: list[float]) -> str:
        """Entry point for camera loop to submit one face embedding."""
        if not self._camera_identity:
            return "camera-disabled"
        event = self._camera_identity.observe_embedding(embedding)
        if event.status == "recognized" and event.person_id and event.name:
            self.set_recognized_identity_context(
                person_id=event.person_id,
                name=event.name,
                confidence=event.confidence,
                face_embeddings_ref=self._settings.camera_identity_store_path,
            )
        return event.status

    def observe_camera_frame(self, image_bytes: bytes) -> list[str]:
        """Entry point for camera loop to submit one encoded frame."""
        if not self._camera_identity:
            return []
        statuses: list[str] = []
        for event in self._camera_identity.process_frame(image_bytes):
            statuses.append(event.status)
            if event.status == "recognized" and event.person_id and event.name:
                self.set_recognized_identity_context(
                    person_id=event.person_id,
                    name=event.name,
                    confidence=event.confidence,
                    face_embeddings_ref=self._settings.camera_identity_store_path,
                )
        return statuses

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

    @staticmethod
    def _needs_fresh_data(user_text: str) -> bool:
        text = user_text.strip().lower()
        freshness_signals = (
            "latest",
            "current",
            "right now",
            "today",
            "now",
            "news",
            "headline",
            "headlines",
            "breaking",
            "internet",
            "web",
            "online",
        )
        return any(k in text for k in freshness_signals)

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
        telemetry: TurnTelemetry | None = None,
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
                started = perf_counter()
                out = await self._mcp.invoke(search_tool, args)
                if telemetry is not None:
                    telemetry.add_tool_call(
                        name=search_tool,
                        latency_ms=(perf_counter() - started) * 1000,
                        args_json=args,
                        result_text=out,
                    )
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
                started = perf_counter()
                out = await self._mcp.invoke(time_tool, args)
                if telemetry is not None:
                    telemetry.add_tool_call(
                        name=time_tool,
                        latency_ms=(perf_counter() - started) * 1000,
                        args_json=args,
                        result_text=out,
                    )
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
                    started = perf_counter()
                    out = await self._mcp.invoke(fetch_tool, args)
                    if telemetry is not None:
                        telemetry.add_tool_call(
                            name=fetch_tool,
                            latency_ms=(perf_counter() - started) * 1000,
                            args_json=args,
                            result_text=out,
                        )
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
        on_telemetry: Callable[[TurnTelemetry], None] | None = None,
    ) -> tuple[list[dict[str, Any]], str]:
        turn_start = perf_counter()
        telemetry = TurnTelemetry()

        if self._camera_identity and self._camera_identity.name_prompt_pending():
            if self._camera_identity.pop_name_prompt():
                prompt_text = self._settings.camera_unknown_prompt
                history_with_prompt = history + [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": prompt_text},
                ]
                try:
                    self._memory.ingest_turn(user_text, prompt_text)
                except Exception:
                    log.exception("memory.ingest_failed")
                telemetry.total_turn_latency_ms = (perf_counter() - turn_start) * 1000
                if on_telemetry:
                    on_telemetry(telemetry)
                return history_with_prompt, prompt_text

        if self._camera_identity:
            enrolled = self._camera_identity.consume_enrollment_name(user_text)
            if enrolled is not None:
                self.set_recognized_identity_context(
                    person_id=enrolled.person_id,
                    name=enrolled.name,
                    confidence=1.0,
                    face_embeddings_ref=self._settings.camera_identity_store_path,
                )
                self._memory.upsert_user_profile(name=enrolled.name)
                user_text = f"My name is {enrolled.name}."

        packed_history, recent_history = self._pack_old_history(history)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.build_system_prompt()},
        ]
        if packed_history:
            messages.append(packed_history)
        messages.extend(recent_history)
        messages.append({"role": "user", "content": user_text})
        tools_payload = [t.to_openai() for t in self._mcp.tools] if self._mcp.tools else None
        force_tool_first_round = bool(tools_payload and self._needs_fresh_data(user_text))
        telemetry.prep_latency_ms = (perf_counter() - turn_start) * 1000
        for round_i in range(self._settings.max_tool_rounds):
            log.debug("agent.llm_round", round=round_i, num_tools=len(self._mcp.tools))
            round_input_chars = estimate_message_chars(messages)
            round_input_tokens = estimate_message_tokens(messages)
            llm_start = perf_counter()
            tool_choice: str | dict[str, Any] | None = "required" if (force_tool_first_round and round_i == 0) else None
            if self._settings.stream_responses:
                assistant = await self._llm.stream_complete(
                    messages,
                    tools=tools_payload if tools_payload else None,
                    tool_choice=tool_choice,
                    on_text_delta=on_text_delta,
                )
            else:
                resp = await self._llm.chat(
                    messages,
                    tools=tools_payload if tools_payload else None,
                    tool_choice=tool_choice,
                )
                assistant = _completion_assistant_as_dict(resp.choices[0].message)
            llm_latency_ms = (perf_counter() - llm_start) * 1000
            round_output_chars = len((assistant.get("content") or "").strip())
            if assistant.get("tool_calls"):
                round_output_chars += len(json.dumps(assistant["tool_calls"], ensure_ascii=False))
            telemetry.add_llm_round(
                round_index=round_i,
                input_chars=round_input_chars,
                input_tokens_est=round_input_tokens,
                output_chars=round_output_chars,
                output_tokens_est=estimate_tokens_from_text((assistant.get("content") or "").strip())
                + (
                    estimate_tokens_from_text(json.dumps(assistant["tool_calls"], ensure_ascii=False))
                    if assistant.get("tool_calls")
                    else 0
                ),
                latency_ms=llm_latency_ms,
            )

            tool_calls = assistant.get("tool_calls")
            if tool_calls:
                if not self._mcp.tools:
                    log.warning("agent.unexpected_tool_calls", round=round_i)
                    telemetry.total_turn_latency_ms = (perf_counter() - turn_start) * 1000
                    if on_telemetry:
                        on_telemetry(telemetry)
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
                    tool_start = perf_counter()
                    out = await self._mcp.invoke(name, args)
                    telemetry.add_tool_call(
                        name=name,
                        latency_ms=(perf_counter() - tool_start) * 1000,
                        args_json=args,
                        result_text=out,
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.get("id", ""),
                            "content": out,
                        }
                    )
                continue

            # Safety fallback for explicit command-style tool requests.
            if round_i == 0 and tools_payload and not tool_calls and (
                self._is_explicit_tool_command(user_text) or self._needs_fresh_data(user_text)
            ):
                fallback_msgs = await self._heuristic_tool_fallback(
                    user_text,
                    on_tool_round_start=on_tool_round_start,
                    on_tool_call=on_tool_call,
                    telemetry=telemetry,
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
            telemetry.total_turn_latency_ms = (perf_counter() - turn_start) * 1000
            if on_telemetry:
                on_telemetry(telemetry)
            return new_history, text

        telemetry.total_turn_latency_ms = (perf_counter() - turn_start) * 1000
        if on_telemetry:
            on_telemetry(telemetry)
        return (
            history + [{"role": "user", "content": user_text}],
            "Stopped: max tool rounds exceeded.",
        )
