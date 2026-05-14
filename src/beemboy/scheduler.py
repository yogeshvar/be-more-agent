from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from .storage import Store


@dataclass(frozen=True)
class ReminderEvent:
    reminder_id: int
    title: str
    remind_at: str


@dataclass(frozen=True)
class AlarmEvent:
    alarm_id: int
    alarm_time: str


class ReminderScheduler:
    def __init__(self, store: Store, on_reminder: Callable[[ReminderEvent], None]):
        self.store = store
        self.on_reminder = on_reminder
        self.on_alarm: Callable[[AlarmEvent], None] = lambda _event: None
        self._last_alarm_key: str | None = None

    def poll_once(self, now: datetime | None = None) -> tuple[list[ReminderEvent], list[AlarmEvent]]:
        now = now or datetime.now()
        due = self.store.pending_reminders(now.isoformat(timespec="seconds"))
        reminder_events: list[ReminderEvent] = []
        for reminder in due:
            reminder_event = ReminderEvent(
                reminder_id=reminder.id,
                title=reminder.title,
                remind_at=reminder.remind_at,
            )
            self.on_reminder(reminder_event)
            self.store.mark_reminder_done(reminder.id)
            self.store.log_event("reminder_fired", f"{reminder.id}:{reminder.title}")
            reminder_events.append(reminder_event)

        alarm_events: list[AlarmEvent] = []
        minute_key = now.strftime("%Y-%m-%dT%H:%M")
        if minute_key != self._last_alarm_key:
            current_hm = now.strftime("%H:%M")
            for alarm_id, alarm_time, _recurrence in self.store.active_alarms():
                if alarm_time == current_hm:
                    alarm_event = AlarmEvent(alarm_id=alarm_id, alarm_time=alarm_time)
                    self.on_alarm(alarm_event)
                    self.store.log_event("alarm_fired", f"{alarm_id}:{alarm_time}")
                    alarm_events.append(alarm_event)
            self._last_alarm_key = minute_key

        return reminder_events, alarm_events
