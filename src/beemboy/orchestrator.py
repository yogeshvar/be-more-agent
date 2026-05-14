from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .context_loader import load_context_bundle
from .guardrails import check_guardrails
from .intent import parse_intent
from .llm import LlamaClient
from .scheduler import ReminderScheduler
from .storage import Store


@dataclass
class Orchestrator:
    store: Store
    llm: LlamaClient
    scheduler: ReminderScheduler
    context_dir: str
    enable_tts: bool = False

    def handle_user_text(self, text: str) -> str:
        guardrail = check_guardrails(text)
        if guardrail.blocked:
            self.store.log_event("guardrail_block", guardrail.reason or "unknown")
            return guardrail.response or "Request blocked by policy."

        intent = parse_intent(text)
        if intent.name == "diary_write":
            entry_id = self.store.add_diary_entry(intent.slots["content"])
            return f"Saved diary entry #{entry_id}."

        if intent.name == "diary_read_date":
            entries = self.store.diary_for_date(intent.slots["date"])
            if not entries:
                return f"No diary entries for {intent.slots['date']}."
            return "\n".join(f"- {entry}" for entry in entries)

        if intent.name == "important_date_add":
            important_id = self.store.add_important_date(
                name=intent.slots["name"],
                event_date=intent.slots["date"],
                lead_time_days=int(intent.slots["lead_days"]),
            )
            return (
                f"Saved important date #{important_id}: {intent.slots['name']} on "
                f"{intent.slots['date']}."
            )

        if intent.name == "important_date_list":
            items = self.store.list_important_dates()
            if not items:
                return "No important dates saved."
            return "\n".join(
                f"- #{item_id}: {name} on {event_date} ({recurrence}, lead {lead_days}d)"
                for item_id, name, event_date, recurrence, lead_days in items
            )

        if intent.name == "events_on_date":
            date = intent.slots["date"]
            diary_entries = self.store.diary_for_date(date)
            special_dates = self.store.important_dates_on(date)
            if not diary_entries and not special_dates:
                return f"No diary entries or important dates for {date}."
            lines = [f"For {date}:"]
            if special_dates:
                lines.append("Important dates:")
                lines.extend(f"- {name}" for name in special_dates)
            if diary_entries:
                lines.append("Diary entries:")
                lines.extend(f"- {entry}" for entry in diary_entries)
            return "\n".join(lines)

        if intent.name == "reminder_add":
            reminder_id = self.store.add_reminder(
                intent.slots["title"], intent.slots["at"], recurrence="none"
            )
            return f"Reminder #{reminder_id} set for {intent.slots['at']}."

        if intent.name == "alarm_add":
            alarm_id = self.store.add_alarm(intent.slots["time"], recurrence="daily", enabled=True)
            return f"Alarm #{alarm_id} set at {intent.slots['time']} every day."

        if intent.name == "reminder_list":
            reminders = self.store.pending_reminders(datetime.now().isoformat(timespec="seconds"))
            if not reminders:
                return "No pending reminders."
            return "\n".join(f"- #{r.id}: {r.title} at {r.remind_at}" for r in reminders)

        context_bundle = load_context_bundle(self._context_dir_path())
        return self.llm.chat(context_bundle, intent.slots["text"])

    def tick_scheduler(self) -> list[str]:
        fired: list[str] = []

        def _on_reminder(event) -> None:
            fired.append(f"Reminder: {event.title}")

        def _on_alarm(event) -> None:
            fired.append(f"Alarm: It is {event.alarm_time}")

        # Temporary callback swap keeps scheduler reusable and testable.
        self.scheduler.on_reminder = _on_reminder
        self.scheduler.on_alarm = _on_alarm
        self.scheduler.poll_once()
        return fired

    def _context_dir_path(self):
        from pathlib import Path

        return Path(self.context_dir)
