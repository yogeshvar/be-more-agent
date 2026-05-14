from datetime import datetime
from pathlib import Path

from beemboy.scheduler import ReminderScheduler
from beemboy.storage import Store


def test_scheduler_fires_due_reminder(tmp_path: Path):
    store = Store(tmp_path / "assistant.db")
    store.add_reminder("drink water", "2026-05-12T09:00:00")
    fired = []
    scheduler = ReminderScheduler(store, on_reminder=lambda event: fired.append(event.title))
    reminder_events, alarm_events = scheduler.poll_once(now=datetime.fromisoformat("2026-05-12T09:30:00"))
    assert len(reminder_events) == 1
    assert alarm_events == []
    assert fired == ["drink water"]


def test_scheduler_fires_alarm_once_per_minute(tmp_path: Path):
    store = Store(tmp_path / "assistant.db")
    store.add_alarm("07:30")
    alarms = []
    scheduler = ReminderScheduler(store, on_reminder=lambda _event: None)
    scheduler.on_alarm = lambda event: alarms.append(event.alarm_time)
    scheduler.poll_once(now=datetime.fromisoformat("2026-05-12T07:30:01"))
    scheduler.poll_once(now=datetime.fromisoformat("2026-05-12T07:30:40"))
    assert alarms == ["07:30"]
