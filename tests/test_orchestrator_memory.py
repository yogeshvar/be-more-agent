from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from beemboy.agent.orchestrator import AgentOrchestrator
from beemboy.config.settings import Settings


class FakeLLM:
    async def chat(self, messages, tools=None, tool_choice=None):  # noqa: ANN001, ARG002
        msg = SimpleNamespace(content="Nice to meet you, Alex.", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    async def stream_complete(self, messages, tools=None, tool_choice=None, on_text_delta=None):  # noqa: ANN001, ARG002
        return {"role": "assistant", "content": "streamed", "tool_calls": None}


class FakeMCP:
    tools = []

    async def invoke(self, openai_name: str, arguments_json: str) -> str:  # noqa: ARG002
        return ""


class OrchestratorMemoryTests(unittest.TestCase):
    def test_orchestrator_injects_memory_and_persists_turn(self) -> None:
        with TemporaryDirectory() as tmp:
            settings = Settings(
                live_context_enabled=False,
                stream_responses=False,
                memory_store_path=str(Path(tmp) / "memory.json"),
                context_compression=False,
            )
            orch = AgentOrchestrator(settings, FakeLLM(), FakeMCP())
            prompt = orch.build_system_prompt()
            self.assertIn("Memory context block (internal):", prompt)

            history, reply = asyncio.run(orch.run_turn([], "my name is alex and I'm working on pi setup"))
            self.assertEqual(reply, "Nice to meet you, Alex.")
            self.assertEqual(len(history), 2)

            reloaded = AgentOrchestrator(settings, FakeLLM(), FakeMCP())
            summary = reloaded.build_system_prompt()
            self.assertIn("Alex", summary)
