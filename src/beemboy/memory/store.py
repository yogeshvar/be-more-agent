from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()


def _clean_value(value: str, *, max_len: int = 120) -> str:
    cleaned = " ".join(value.strip().split())
    return cleaned[:max_len]


@dataclass
class UserProfile:
    name: str | None = None
    location: str | None = None
    preferences: list[str] = field(default_factory=list)
    goals: list[str] = field(default_factory=list)


@dataclass
class LifeContext:
    routines: list[str] = field(default_factory=list)
    projects: list[str] = field(default_factory=list)
    relationships: dict[str, str] = field(default_factory=dict)


@dataclass
class JournalEntry:
    timestamp: str
    source: str
    content: str


@dataclass
class StockWatchItem:
    symbol: str
    note: str | None = None
    last_mentioned_at: str = field(default_factory=_utc_now_iso)


@dataclass
class KnownIdentity:
    person_id: str
    name: str
    face_embeddings_ref: str | None = None
    first_seen_at: str = field(default_factory=_utc_now_iso)
    last_seen_at: str = field(default_factory=_utc_now_iso)


@dataclass
class MemoryState:
    user_profile: UserProfile = field(default_factory=UserProfile)
    life_context: LifeContext = field(default_factory=LifeContext)
    journal_entries: list[JournalEntry] = field(default_factory=list)
    stock_watchlist: list[StockWatchItem] = field(default_factory=list)
    known_identities: list[KnownIdentity] = field(default_factory=list)


class MemoryStore:
    """Durable local memory backed by a single JSON file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.state = MemoryState()
        self.load()

    def load(self) -> MemoryState:
        if not self.path.is_file():
            self.state = MemoryState()
            return self.state
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.state = MemoryState(
            user_profile=UserProfile(**(data.get("user_profile") or {})),
            life_context=LifeContext(**(data.get("life_context") or {})),
            journal_entries=[JournalEntry(**item) for item in data.get("journal_entries", []) if isinstance(item, dict)],
            stock_watchlist=[StockWatchItem(**item) for item in data.get("stock_watchlist", []) if isinstance(item, dict)],
            known_identities=[KnownIdentity(**item) for item in data.get("known_identities", []) if isinstance(item, dict)],
        )
        return self.state

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self.state)
        self.path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    def upsert_user_profile(self, **kwargs: Any) -> None:
        for key in ("name", "location"):
            value = kwargs.get(key)
            if isinstance(value, str) and value.strip():
                setattr(self.state.user_profile, key, _clean_value(value))
        self._merge_unique_list(self.state.user_profile.preferences, kwargs.get("preferences"), max_len=8)
        self._merge_unique_list(self.state.user_profile.goals, kwargs.get("goals"), max_len=8)
        self.save()

    def upsert_life_context(
        self,
        *,
        routines: list[str] | None = None,
        projects: list[str] | None = None,
        relationships: dict[str, str] | None = None,
    ) -> None:
        self._merge_unique_list(self.state.life_context.routines, routines, max_len=12)
        self._merge_unique_list(self.state.life_context.projects, projects, max_len=12)
        if relationships:
            for name, relation in relationships.items():
                if name.strip() and relation.strip():
                    self.state.life_context.relationships[_clean_value(name, max_len=60)] = _clean_value(
                        relation, max_len=60
                    )
        self.save()

    def append_journal(self, content: str, *, source: str, timestamp: str | None = None) -> None:
        cleaned = _clean_value(content, max_len=500)
        if not cleaned:
            return
        self.state.journal_entries.append(
            JournalEntry(
                timestamp=timestamp or _utc_now_iso(),
                source=source,
                content=cleaned,
            )
        )
        # Keep on-disk size bounded on Pi.
        self.state.journal_entries = self.state.journal_entries[-200:]
        self.save()

    def upsert_stock(self, symbol: str, *, note: str | None = None, mentioned_at: str | None = None) -> None:
        normalized = symbol.strip().upper().lstrip("$")
        if not normalized:
            return
        for item in self.state.stock_watchlist:
            if item.symbol == normalized:
                item.last_mentioned_at = mentioned_at or _utc_now_iso()
                if note:
                    item.note = _clean_value(note, max_len=100)
                self.save()
                return
        self.state.stock_watchlist.append(
            StockWatchItem(
                symbol=normalized,
                note=_clean_value(note, max_len=100) if note else None,
                last_mentioned_at=mentioned_at or _utc_now_iso(),
            )
        )
        self.state.stock_watchlist = self.state.stock_watchlist[-30:]
        self.save()

    def upsert_known_identity(
        self,
        *,
        person_id: str,
        name: str,
        face_embeddings_ref: str | None = None,
        seen_at: str | None = None,
    ) -> None:
        when = seen_at or _utc_now_iso()
        for item in self.state.known_identities:
            if item.person_id == person_id:
                item.name = _clean_value(name, max_len=80)
                item.last_seen_at = when
                if face_embeddings_ref:
                    item.face_embeddings_ref = _clean_value(face_embeddings_ref, max_len=160)
                self.save()
                return
        self.state.known_identities.append(
            KnownIdentity(
                person_id=_clean_value(person_id, max_len=80),
                name=_clean_value(name, max_len=80),
                face_embeddings_ref=_clean_value(face_embeddings_ref, max_len=160) if face_embeddings_ref else None,
                first_seen_at=when,
                last_seen_at=when,
            )
        )
        self.state.known_identities = self.state.known_identities[-50:]
        self.save()

    def ingest_turn(self, user_text: str, assistant_text: str) -> None:
        self.append_journal(user_text, source="user")
        self.append_journal(assistant_text, source="assistant")

        name_match = re.search(r"\bmy name is ([a-z][a-z '\-]{1,48})\b", user_text, flags=re.IGNORECASE)
        if name_match:
            self.upsert_user_profile(name=name_match.group(1).title())

        location_match = re.search(r"\bi live in ([a-z][a-z .'\-]{1,60})\b", user_text, flags=re.IGNORECASE)
        if location_match:
            self.upsert_user_profile(location=location_match.group(1).title())

        routine_match = re.search(r"\b(?:every|usually|often)\s+(.{5,90})", user_text, flags=re.IGNORECASE)
        if routine_match:
            self.upsert_life_context(routines=[routine_match.group(0)])

        project_match = re.search(
            r"\b(?:working on|building|shipping|my project is)\s+(.{3,90})",
            user_text,
            flags=re.IGNORECASE,
        )
        if project_match:
            self.upsert_life_context(projects=[project_match.group(1)])

        rel_match = re.search(
            r"\b([A-Z][a-z]{1,24}) is my (wife|husband|partner|friend|brother|sister|mom|dad)\b",
            user_text,
        )
        if rel_match:
            self.upsert_life_context(relationships={rel_match.group(1): rel_match.group(2)})

        stock_mentions = set(re.findall(r"\$([A-Z]{1,5})\b", user_text))
        for symbol in stock_mentions:
            self.upsert_stock(symbol)

    def summarize_for_prompt(self) -> str:
        profile = self.state.user_profile
        life = self.state.life_context
        journal = self.state.journal_entries[-5:]
        watchlist = self.state.stock_watchlist[-6:]
        identities = self.state.known_identities[-5:]

        chunks: list[str] = ["Memory context (internal):"]
        if profile.name or profile.location or profile.preferences or profile.goals:
            chunks.append(
                "Profile: "
                f"name={profile.name or 'unknown'}, "
                f"location={profile.location or 'unknown'}, "
                f"preferences={profile.preferences or []}, "
                f"goals={profile.goals or []}"
            )
        if life.projects or life.routines or life.relationships:
            chunks.append(
                "Life context: "
                f"projects={life.projects[:4]}, "
                f"routines={life.routines[:4]}, "
                f"relationships={dict(list(life.relationships.items())[:5])}"
            )
        if watchlist:
            chunks.append(
                "Stock watchlist: "
                + ", ".join(item.symbol if not item.note else f"{item.symbol}({item.note})" for item in watchlist)
            )
        if identities:
            chunks.append(
                "Known identities: "
                + ", ".join(f"{item.name}<{item.person_id}> last_seen={item.last_seen_at}" for item in identities)
            )
        if journal:
            snippets = [f"[{entry.source}] {entry.content}" for entry in journal]
            chunks.append("Recent journal: " + " | ".join(snippets))
        return "\n".join(chunks)

    @staticmethod
    def _merge_unique_list(target: list[str], values: Any, *, max_len: int) -> None:
        if not values:
            return
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, list):
            return
        for value in values:
            if not isinstance(value, str):
                continue
            cleaned = _clean_value(value)
            if cleaned and cleaned not in target:
                target.append(cleaned)
        del target[:-max_len]
