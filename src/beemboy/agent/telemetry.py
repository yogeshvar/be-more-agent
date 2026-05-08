from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import ceil
from typing import Any


def estimate_tokens_from_text(text: str) -> int:
    """Cheap offline token estimate using character length heuristic."""
    if not text:
        return 0
    return ceil(len(text) / 4)


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def estimate_message_chars(messages: list[dict[str, Any]]) -> int:
    total = 0
    for message in messages:
        total += len(_stringify(message.get("role")))
        total += len(_stringify(message.get("content")))
        for tool_call in message.get("tool_calls") or []:
            fn = tool_call.get("function") or {}
            total += len(_stringify(fn.get("name")))
            total += len(_stringify(fn.get("arguments")))
    return total


def estimate_message_tokens(messages: list[dict[str, Any]]) -> int:
    return ceil(estimate_message_chars(messages) / 4)


@dataclass(slots=True)
class ToolTelemetry:
    name: str
    latency_ms: float
    args_chars: int
    result_chars: int
    result_tokens_est: int


@dataclass(slots=True)
class LLMRoundTelemetry:
    round_index: int
    input_chars: int
    input_tokens_est: int
    output_chars: int
    output_tokens_est: int
    latency_ms: float


@dataclass(slots=True)
class TurnTelemetry:
    prep_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    total_tool_latency_ms: float = 0.0
    total_turn_latency_ms: float = 0.0
    total_input_chars: int = 0
    total_input_tokens_est: int = 0
    total_output_chars: int = 0
    total_output_tokens_est: int = 0
    llm_rounds: list[LLMRoundTelemetry] = field(default_factory=list)
    tool_calls: list[ToolTelemetry] = field(default_factory=list)

    def add_llm_round(
        self,
        *,
        round_index: int,
        input_chars: int,
        input_tokens_est: int,
        output_chars: int,
        output_tokens_est: int,
        latency_ms: float,
    ) -> None:
        self.total_input_chars += input_chars
        self.total_input_tokens_est += input_tokens_est
        self.total_output_chars += output_chars
        self.total_output_tokens_est += output_tokens_est
        self.llm_latency_ms += latency_ms
        self.llm_rounds.append(
            LLMRoundTelemetry(
                round_index=round_index,
                input_chars=input_chars,
                input_tokens_est=input_tokens_est,
                output_chars=output_chars,
                output_tokens_est=output_tokens_est,
                latency_ms=latency_ms,
            )
        )

    def add_tool_call(self, *, name: str, latency_ms: float, args_json: str, result_text: str) -> None:
        result_tokens_est = estimate_tokens_from_text(result_text)
        self.total_tool_latency_ms += latency_ms
        self.tool_calls.append(
            ToolTelemetry(
                name=name,
                latency_ms=latency_ms,
                args_chars=len(args_json or ""),
                result_chars=len(result_text or ""),
                result_tokens_est=result_tokens_est,
            )
        )

    def to_debug_dict(self) -> dict[str, Any]:
        return {
            "input": {
                "chars_total": self.total_input_chars,
                "tokens_est_total": self.total_input_tokens_est,
            },
            "output": {
                "chars_total": self.total_output_chars,
                "tokens_est_total": self.total_output_tokens_est,
            },
            "latency_ms": {
                "prep": round(self.prep_latency_ms, 2),
                "llm_total": round(self.llm_latency_ms, 2),
                "tools_total": round(self.total_tool_latency_ms, 2),
                "turn_total": round(self.total_turn_latency_ms, 2),
            },
            "llm_rounds": [asdict(r) for r in self.llm_rounds],
            "tool_calls": [asdict(t) for t in self.tool_calls],
        }


@dataclass(slots=True)
class VoiceTurnTelemetry:
    stt_latency_ms: float = 0.0
    tts_latency_ms: float = 0.0
    user_audio_ms: float = 0.0

    def to_debug_dict(self) -> dict[str, Any]:
        return {
            "stt_latency_ms": round(self.stt_latency_ms, 2),
            "tts_latency_ms": round(self.tts_latency_ms, 2),
            "user_audio_ms": round(self.user_audio_ms, 2),
        }
