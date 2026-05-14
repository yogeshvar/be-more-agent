from pathlib import Path

from beemboy.storage import Store


def test_diary_roundtrip(tmp_path: Path):
    store = Store(tmp_path / "assistant.db")
    store.add_diary_entry("today was great", entry_date="2026-05-12")
    entries = store.diary_for_date("2026-05-12")
    assert entries == ["today was great"]


def test_reminder_lifecycle(tmp_path: Path):
    store = Store(tmp_path / "assistant.db")
    rid = store.add_reminder("pay rent", "2026-05-12T09:00:00")
    due = store.pending_reminders("2026-05-12T10:00:00")
    assert len(due) == 1
    assert due[0].id == rid
    store.mark_reminder_done(rid)
    after = store.pending_reminders("2026-05-12T10:05:00")
    assert after == []
