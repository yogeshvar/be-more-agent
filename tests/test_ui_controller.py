from __future__ import annotations

import unittest

from beemboy.ui.controller import UIController


class FakeOrchestrator:
    def __init__(self) -> None:
        self.saved = 0
        self.shutdown_called = 0
        self.frames_seen = 0

    def list_known_identities(self) -> list[dict[str, str]]:
        return [
            {
                "person_id": "person-1",
                "name": "Mags",
                "source": "registry",
                "last_seen_at": "2026-01-01T00:00:00+00:00",
            }
        ]

    def observe_camera_frame(self, image_bytes: bytes) -> list[str]:
        self.frames_seen += 1
        return ["recognized"] if image_bytes else []

    def save_state(self) -> None:
        self.saved += 1

    def shutdown(self) -> None:
        self.shutdown_called += 1


class UIControllerTests(unittest.TestCase):
    def test_save_now_flushes_state(self) -> None:
        orchestrator = FakeOrchestrator()
        controller = UIController(orchestrator)
        controller.save_now()
        self.assertEqual(orchestrator.saved, 1)
        self.assertIn("Saved memory and identity registry.", controller.get_status_text())

    def test_safe_exit_stops_and_shutdowns(self) -> None:
        orchestrator = FakeOrchestrator()
        controller = UIController(orchestrator)
        controller.start()
        controller.safe_exit()
        self.assertFalse(controller.running)
        self.assertEqual(orchestrator.shutdown_called, 1)
        status = controller.get_status_text()
        self.assertIn("Camera loop stopped.", status)
        self.assertIn("Safe exit complete. State flushed.", status)

    def test_process_frame_only_when_running(self) -> None:
        orchestrator = FakeOrchestrator()
        controller = UIController(orchestrator)
        self.assertEqual(controller.process_frame(b"frame"), [])
        self.assertEqual(orchestrator.frames_seen, 0)
        controller.start()
        out = controller.process_frame(b"frame")
        self.assertEqual(out, ["recognized"])
        self.assertEqual(orchestrator.frames_seen, 1)

    def test_list_identities_maps_fields(self) -> None:
        orchestrator = FakeOrchestrator()
        controller = UIController(orchestrator)
        items = controller.list_identities()
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].name, "Mags")
        self.assertEqual(items[0].person_id, "person-1")
