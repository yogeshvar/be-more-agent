from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from beemboy.vision.embedder import cosine_similarity


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()


def _clean_name(name: str) -> str:
    compact = " ".join(name.strip().split())
    compact = "".join(ch for ch in compact if ch.isalpha() or ch in {" ", "-", "'"})
    return compact[:80].strip().title()


@dataclass(slots=True)
class IdentityRecord:
    person_id: str
    name: str
    embeddings: list[list[float]] = field(default_factory=list)
    first_seen_at: str = field(default_factory=_utc_now_iso)
    last_seen_at: str = field(default_factory=_utc_now_iso)


@dataclass(slots=True)
class RegistryMatch:
    matched: bool
    person_id: str | None = None
    name: str | None = None
    confidence: float = 0.0


class FaceRegistry:
    """Persistent identity registry storing embeddings only."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self._records: list[IdentityRecord] = []
        self.load()

    @property
    def records(self) -> list[IdentityRecord]:
        return list(self._records)

    def load(self) -> list[IdentityRecord]:
        if not self.path.is_file():
            self._records = []
            return self._records
        data = json.loads(self.path.read_text(encoding="utf-8"))
        items = data.get("identities", []) if isinstance(data, dict) else []
        out: list[IdentityRecord] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            rec = IdentityRecord(
                person_id=str(item.get("person_id") or ""),
                name=_clean_name(str(item.get("name") or "")),
                embeddings=[
                    [float(v) for v in emb]
                    for emb in item.get("embeddings", [])
                    if isinstance(emb, list) and emb
                ],
                first_seen_at=str(item.get("first_seen_at") or _utc_now_iso()),
                last_seen_at=str(item.get("last_seen_at") or _utc_now_iso()),
            )
            if rec.person_id and rec.name and rec.embeddings:
                out.append(rec)
        self._records = out
        return self._records

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {"identities": [asdict(item) for item in self._records]}
        self.path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    def match(self, embedding: list[float], *, threshold: float) -> RegistryMatch:
        if not embedding:
            return RegistryMatch(matched=False, confidence=0.0)
        best_rec: IdentityRecord | None = None
        best_score = -1.0
        for rec in self._records:
            for known in rec.embeddings:
                score = cosine_similarity(embedding, known)
                if score > best_score:
                    best_score = score
                    best_rec = rec
        if best_rec is None or best_score < threshold:
            return RegistryMatch(matched=False, confidence=max(best_score, 0.0))
        best_rec.last_seen_at = _utc_now_iso()
        self.save()
        return RegistryMatch(
            matched=True,
            person_id=best_rec.person_id,
            name=best_rec.name,
            confidence=best_score,
        )

    def enroll(
        self,
        *,
        name: str,
        embedding: list[float],
        person_id: str | None = None,
    ) -> IdentityRecord:
        clean_name = _clean_name(name)
        if not clean_name:
            raise ValueError("Enrollment name is empty after normalization")
        if not embedding:
            raise ValueError("Embedding cannot be empty")
        person_id = person_id or f"person-{uuid4().hex[:12]}"
        now = _utc_now_iso()
        rec = IdentityRecord(
            person_id=person_id,
            name=clean_name,
            embeddings=[embedding],
            first_seen_at=now,
            last_seen_at=now,
        )
        self._records.append(rec)
        self._records = self._records[-120:]
        self.save()
        return rec
