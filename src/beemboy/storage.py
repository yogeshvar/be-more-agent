from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sqlite3


SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS diary_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    entry_date TEXT NOT NULL,
    content TEXT NOT NULL,
    tags TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS important_dates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    event_date TEXT NOT NULL,
    recurrence TEXT NOT NULL DEFAULT 'yearly',
    lead_time_days INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS reminders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    remind_at TEXT NOT NULL,
    recurrence TEXT NOT NULL DEFAULT 'none',
    status TEXT NOT NULL DEFAULT 'scheduled'
);

CREATE TABLE IF NOT EXISTS alarms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    alarm_time TEXT NOT NULL,
    recurrence TEXT NOT NULL DEFAULT 'daily',
    enabled INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS events_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class Reminder:
    id: int
    title: str
    remind_at: str
    recurrence: str
    status: str


class Store:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def add_diary_entry(self, content: str, entry_date: str | None = None, tags: str = "") -> int:
        if not entry_date:
            entry_date = datetime.now().date().isoformat()
        created_at = datetime.now().isoformat(timespec="seconds")
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO diary_entries(created_at, entry_date, content, tags) VALUES (?, ?, ?, ?)",
                (created_at, entry_date, content.strip(), tags.strip()),
            )
            return int(cur.lastrowid)

    def diary_for_date(self, entry_date: str) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT content FROM diary_entries WHERE entry_date = ? ORDER BY id ASC",
                (entry_date,),
            ).fetchall()
            return [r["content"] for r in rows]

    def add_important_date(
        self, name: str, event_date: str, recurrence: str = "yearly", lead_time_days: int = 1
    ) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO important_dates(name, event_date, recurrence, lead_time_days) VALUES (?, ?, ?, ?)",
                (name.strip(), event_date, recurrence, lead_time_days),
            )
            return int(cur.lastrowid)

    def list_important_dates(self) -> list[tuple[int, str, str, str, int]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, name, event_date, recurrence, lead_time_days FROM important_dates ORDER BY event_date ASC"
            ).fetchall()
            return [
                (
                    int(r["id"]),
                    r["name"],
                    r["event_date"],
                    r["recurrence"],
                    int(r["lead_time_days"]),
                )
                for r in rows
            ]

    def important_dates_on(self, event_date: str) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT name FROM important_dates WHERE event_date = ? ORDER BY id ASC", (event_date,)
            ).fetchall()
            return [r["name"] for r in rows]

    def add_reminder(self, title: str, remind_at: str, recurrence: str = "none") -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO reminders(title, remind_at, recurrence, status) VALUES (?, ?, ?, 'scheduled')",
                (title.strip(), remind_at, recurrence),
            )
            return int(cur.lastrowid)

    def pending_reminders(self, now_iso: str) -> list[Reminder]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM reminders WHERE status='scheduled' AND remind_at <= ? ORDER BY remind_at ASC",
                (now_iso,),
            ).fetchall()
            return [
                Reminder(
                    id=int(r["id"]),
                    title=r["title"],
                    remind_at=r["remind_at"],
                    recurrence=r["recurrence"],
                    status=r["status"],
                )
                for r in rows
            ]

    def mark_reminder_done(self, reminder_id: int) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE reminders SET status='done' WHERE id = ?", (reminder_id,))

    def add_alarm(self, alarm_time: str, recurrence: str = "daily", enabled: bool = True) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO alarms(alarm_time, recurrence, enabled) VALUES (?, ?, ?)",
                (alarm_time, recurrence, 1 if enabled else 0),
            )
            return int(cur.lastrowid)

    def active_alarms(self) -> list[tuple[int, str, str]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, alarm_time, recurrence FROM alarms WHERE enabled = 1 ORDER BY id ASC"
            ).fetchall()
            return [(int(r["id"]), r["alarm_time"], r["recurrence"]) for r in rows]

    def log_event(self, event_type: str, payload: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO events_log(created_at, event_type, payload) VALUES (?, ?, ?)",
                (datetime.now().isoformat(timespec="seconds"), event_type, payload),
            )
