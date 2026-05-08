from __future__ import annotations

import json
import unittest

from beemboy.memory.store import MemoryStore


class MemoryStoreTests(unittest.TestCase):
    def test_memory_store_persists_and_summarizes(self) -> None:
        with self.subTest("persist and summarize"):
            from tempfile import TemporaryDirectory

            with TemporaryDirectory() as tmp:
                from pathlib import Path

                path = Path(tmp) / "memory.json"
                store = MemoryStore(path)
                store.upsert_user_profile(name="Mags", location="Brooklyn")
                store.upsert_life_context(projects=["beemboy v1"], routines=["every morning run"])
                store.upsert_stock("AAPL", note="watch earnings")
                store.upsert_known_identity(person_id="person-1", name="Yogi")
                store.append_journal("Today I shipped memory foundation", source="user")

                reloaded = MemoryStore(path)
                summary = reloaded.summarize_for_prompt()

                self.assertIn("Mags", summary)
                self.assertIn("beemboy v1", summary)
                self.assertIn("AAPL", summary)
                self.assertIn("Yogi<person-1>", summary)

                payload = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(payload["user_profile"]["name"], "Mags")
                self.assertEqual(payload["life_context"]["projects"], ["beemboy v1"])
