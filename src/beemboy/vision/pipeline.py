from __future__ import annotations

from dataclasses import dataclass
import re

from beemboy.vision.detector import FaceDetector
from beemboy.vision.embedder import FaceEmbedder
from beemboy.vision.registry import FaceRegistry, IdentityRecord, RegistryMatch


@dataclass(slots=True)
class IdentityEvent:
    status: str
    person_id: str | None = None
    name: str | None = None
    confidence: float = 0.0


@dataclass(slots=True)
class _PendingEnrollment:
    embedding: list[float]
    prompt_ready: bool = True
    awaiting_name: bool = False


class CameraIdentityPipeline:
    """Camera identity flow: detect -> embed -> match -> enroll."""

    def __init__(
        self,
        *,
        registry: FaceRegistry,
        embedder: FaceEmbedder | None = None,
        detector: FaceDetector | None = None,
        match_threshold: float = 0.82,
    ) -> None:
        self._registry = registry
        self._embedder = embedder or FaceEmbedder()
        self._detector = detector or FaceDetector()
        self._match_threshold = max(0.0, min(1.0, match_threshold))
        self._pending: _PendingEnrollment | None = None

    def process_frame(self, image_bytes: bytes) -> list[IdentityEvent]:
        events: list[IdentityEvent] = []
        for face in self._detector.detect(image_bytes):
            emb = self._embedder.embed(face.crop_bytes)
            events.append(self.observe_embedding(emb))
        return events

    def observe_embedding(self, embedding: list[float]) -> IdentityEvent:
        matched = self._registry.match(embedding, threshold=self._match_threshold)
        if matched.matched:
            self._pending = None
            return IdentityEvent(
                status="recognized",
                person_id=matched.person_id,
                name=matched.name,
                confidence=matched.confidence,
            )
        self._pending = _PendingEnrollment(embedding=embedding, prompt_ready=True, awaiting_name=False)
        return IdentityEvent(status="unknown", confidence=matched.confidence)

    def name_prompt_pending(self) -> bool:
        return bool(self._pending and self._pending.prompt_ready)

    def pop_name_prompt(self) -> bool:
        if not self._pending or not self._pending.prompt_ready:
            return False
        self._pending.prompt_ready = False
        self._pending.awaiting_name = True
        return True

    def consume_enrollment_name(self, utterance: str) -> IdentityRecord | None:
        if not self._pending or not self._pending.awaiting_name:
            return None
        candidate = _extract_name(utterance)
        if not candidate:
            candidate = "Friend"
        pending = self._pending
        self._pending = None
        return self._registry.enroll(name=candidate, embedding=pending.embedding)

    def force_match(self, embedding: list[float]) -> RegistryMatch:
        return self._registry.match(embedding, threshold=self._match_threshold)

    def save_registry(self) -> None:
        self._registry.save()

    def list_identities(self) -> list[IdentityRecord]:
        return self._registry.records


def _extract_name(text: str) -> str:
    t = " ".join(text.strip().split())
    if not t:
        return ""
    patterns = (
        r"(?i)\bmy name is ([a-z][a-z '\-]{0,48})\b",
        r"(?i)\bi am ([a-z][a-z '\-]{0,48})\b",
        r"(?i)\bit'?s ([a-z][a-z '\-]{0,48})\b",
        r"(?i)\bthis is ([a-z][a-z '\-]{0,48})\b",
    )
    for pattern in patterns:
        m = re.search(pattern, t)
        if m:
            return " ".join(m.group(1).split()).title()
    compact = "".join(ch for ch in t if ch.isalpha() or ch in {" ", "-", "'"}).strip()
    if not compact:
        return ""
    return " ".join(compact.split()[:4]).title()
