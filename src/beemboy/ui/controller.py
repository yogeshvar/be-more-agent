from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


class IdentitySource(Protocol):
    def list_known_identities(self) -> list[dict[str, str]]:
        ...

    def observe_camera_frame(self, image_bytes: bytes) -> list[str]:
        ...

    def save_state(self) -> None:
        ...

    def shutdown(self) -> None:
        ...


@dataclass(slots=True)
class IdentityView:
    person_id: str
    name: str
    source: str
    last_seen_at: str


class UIController:
    """Non-GUI controller for desktop camera + persistence actions."""

    def __init__(self, orchestrator: IdentitySource, *, max_status_events: int = 120) -> None:
        self._orchestrator = orchestrator
        self._running = False
        self._status_events: deque[str] = deque(maxlen=max(20, max_status_events))

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._running:
            self._push_status("Camera loop already running.")
            return
        self._running = True
        self._push_status("Camera loop started.")

    def stop(self) -> None:
        if not self._running:
            self._push_status("Camera loop already stopped.")
            return
        self._running = False
        self._push_status("Camera loop stopped.")

    def process_frame(self, jpeg_bytes: bytes) -> list[str]:
        if not self._running:
            return []
        statuses = self._orchestrator.observe_camera_frame(jpeg_bytes)
        if not statuses:
            return []
        self._push_status(f"Recognition events: {', '.join(statuses)}")
        return statuses

    def save_now(self) -> None:
        self._orchestrator.save_state()
        self._push_status("Saved memory and identity registry.")

    def safe_exit(self) -> None:
        self.stop()
        self._orchestrator.shutdown()
        self._push_status("Safe exit complete. State flushed.")

    def list_identities(self) -> list[IdentityView]:
        items = self._orchestrator.list_known_identities()
        return [
            IdentityView(
                person_id=str(item.get("person_id", "")),
                name=str(item.get("name", "Unknown")),
                source=str(item.get("source", "unknown")),
                last_seen_at=str(item.get("last_seen_at", "")),
            )
            for item in items
        ]

    def get_status_text(self) -> str:
        if not self._status_events:
            return "No recognition events yet."
        return "\n".join(self._status_events)

    def _push_status(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._status_events.append(f"[{timestamp}] {message}")
