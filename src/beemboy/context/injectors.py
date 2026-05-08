from __future__ import annotations

from time import monotonic
from typing import Protocol, runtime_checkable

from beemboy.config.settings import Settings
from beemboy.context.live_fetch import build_live_context_block


@runtime_checkable
class SystemContextInjector(Protocol):
    def inject(self, settings: Settings) -> str | None: ...


class ClockInjector:
    """Injects current local time into the system prompt."""

    def inject(self, settings: Settings) -> str | None:  # noqa: ARG002
        from datetime import datetime

        now = datetime.now().astimezone()
        return (
            f"Device local time (trust this for date/time questions): "
            f"{now.strftime('%A, %B %d, %Y at %H:%M:%S %Z')}."
        )


class LiveContextInjector:
    """Optional weather + headlines block (ported from the old shell helper)."""

    def __init__(self) -> None:
        self._cache_value: str | None = None
        self._cache_expires_at: float = 0.0

    def inject(self, settings: Settings) -> str | None:
        now = monotonic()
        if now < self._cache_expires_at:
            return self._cache_value
        self._cache_value = build_live_context_block(settings)
        ttl_s = max(5, int(settings.live_context_ttl_s))
        self._cache_expires_at = now + ttl_s
        return self._cache_value
