from __future__ import annotations

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from beemboy.agent.orchestrator import AgentOrchestrator
from beemboy.config.settings import Settings
from beemboy.vision.registry import FaceRegistry


class EchoLLM:
    async def chat(self, messages, tools=None, tool_choice=None):  # noqa: ANN001, ARG002
        user_msg = next((m for m in reversed(messages) if m.get("role") == "user"), {"content": ""})
        msg = SimpleNamespace(content=f"echo:{user_msg.get('content', '')}", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    async def stream_complete(self, messages, tools=None, tool_choice=None, on_text_delta=None):  # noqa: ANN001, ARG002
        return {"role": "assistant", "content": "streamed", "tool_calls": None}


class FakeMCP:
    tools = []

    async def invoke(self, openai_name: str, arguments_json: str) -> str:  # noqa: ARG002
        return ""


class CameraIdentityTests(unittest.TestCase):
    def test_registry_threshold_matching(self) -> None:
        with TemporaryDirectory() as tmp:
            registry = FaceRegistry(Path(tmp) / "ids.json")
            registry.enroll(name="Mags", embedding=[1.0, 0.0, 0.0], person_id="person-mags")

            strong = registry.match([0.99, 0.01, 0.0], threshold=0.8)
            self.assertTrue(strong.matched)
            self.assertEqual(strong.person_id, "person-mags")

            weak = registry.match([0.2, 0.98, 0.0], threshold=0.8)
            self.assertFalse(weak.matched)

    def test_unknown_then_enrolls_on_next_utterance(self) -> None:
        with TemporaryDirectory() as tmp:
            memory_path = Path(tmp) / "memory.json"
            ids_path = Path(tmp) / "identities.json"
            settings = Settings(
                live_context_enabled=False,
                stream_responses=False,
                context_compression=False,
                camera_enabled=True,
                camera_identity_store_path=str(ids_path),
                memory_store_path=str(memory_path),
                camera_match_threshold=0.8,
            )
            orch = AgentOrchestrator(settings, EchoLLM(), FakeMCP())
            status = orch.observe_camera_embedding([0.0, 1.0, 0.0])
            self.assertEqual(status, "unknown")

            history, reply = asyncio.run(orch.run_turn([], "hey there"))
            self.assertIn("what's your name", reply.lower())
            self.assertEqual(len(history), 2)

            history, second_reply = asyncio.run(orch.run_turn(history, "my name is alex"))
            self.assertEqual(second_reply, "echo:My name is Alex.")

            memory_payload = json.loads(memory_path.read_text(encoding="utf-8"))
            self.assertEqual(memory_payload["user_profile"]["name"], "Alex")
            self.assertEqual(memory_payload["known_identities"][0]["name"], "Alex")
            self.assertEqual(memory_payload["known_identities"][0]["face_embeddings_ref"], str(ids_path))

            ids_payload = json.loads(ids_path.read_text(encoding="utf-8"))
            self.assertIn("identities", ids_payload)
            self.assertIn("embeddings", ids_payload["identities"][0])
            self.assertNotIn("image", ids_payload["identities"][0])

    def test_known_match_sets_recognized_context(self) -> None:
        with TemporaryDirectory() as tmp:
            ids_path = Path(tmp) / "identities.json"
            registry = FaceRegistry(ids_path)
            registry.enroll(name="Jordan", embedding=[1.0, 0.0, 0.0], person_id="person-jordan")
            settings = Settings(
                live_context_enabled=False,
                stream_responses=False,
                context_compression=False,
                camera_enabled=True,
                camera_identity_store_path=str(ids_path),
                camera_match_threshold=0.8,
            )
            orch = AgentOrchestrator(settings, EchoLLM(), FakeMCP())
            status = orch.observe_camera_embedding([1.0, 0.0, 0.0])
            self.assertEqual(status, "recognized")
            prompt = orch.build_system_prompt()
            self.assertIn("Recognized identity context", prompt)
            self.assertIn("Jordan", prompt)
