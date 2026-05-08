from __future__ import annotations

import unittest

from beemboy.agent.telemetry import (
    TurnTelemetry,
    estimate_message_chars,
    estimate_message_tokens,
    estimate_tokens_from_text,
)


class TelemetryTests(unittest.TestCase):
    def test_estimate_tokens_from_text_uses_char_heuristic(self) -> None:
        self.assertEqual(estimate_tokens_from_text(""), 0)
        self.assertEqual(estimate_tokens_from_text("abcd"), 1)
        self.assertEqual(estimate_tokens_from_text("abcde"), 2)

    def test_estimate_message_counts_include_tool_calls(self) -> None:
        messages = [
            {"role": "system", "content": "You are Beemboy"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "time__get_current_time", "arguments": '{"timezone":"UTC"}'},
                    }
                ],
            },
        ]
        chars = estimate_message_chars(messages)
        self.assertGreater(chars, 0)
        self.assertEqual(estimate_message_tokens(messages), (chars + 3) // 4)

    def test_turn_telemetry_aggregates_rounds_and_tools(self) -> None:
        telemetry = TurnTelemetry(prep_latency_ms=7.5)
        telemetry.add_llm_round(
            round_index=0,
            input_chars=120,
            input_tokens_est=30,
            output_chars=44,
            output_tokens_est=11,
            latency_ms=88.0,
        )
        telemetry.add_tool_call(
            name="fetch__fetch",
            latency_ms=19.0,
            args_json='{"url":"https://example.com"}',
            result_text="ok",
        )
        telemetry.total_turn_latency_ms = 144.0

        payload = telemetry.to_debug_dict()
        self.assertEqual(payload["input"]["tokens_est_total"], 30)
        self.assertEqual(payload["output"]["tokens_est_total"], 11)
        self.assertEqual(payload["latency_ms"]["prep"], 7.5)
        self.assertEqual(payload["latency_ms"]["llm_total"], 88.0)
        self.assertEqual(payload["latency_ms"]["tools_total"], 19.0)
        self.assertEqual(payload["latency_ms"]["turn_total"], 144.0)
        self.assertEqual(payload["llm_rounds"][0]["round_index"], 0)
        self.assertEqual(payload["tool_calls"][0]["name"], "fetch__fetch")

