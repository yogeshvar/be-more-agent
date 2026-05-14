from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re


@dataclass(frozen=True)
class Intent:
    name: str
    slots: dict[str, str]


def parse_intent(text: str) -> Intent:
    t = text.strip()
    lower = t.lower()

    if lower.startswith("diary:"):
        return Intent("diary_write", {"content": t.split(":", 1)[1].strip()})

    match_diary_date = re.search(r"\bdiary\s+on\s+(\d{4}-\d{2}-\d{2})\b", lower)
    if match_diary_date:
        return Intent("diary_read_date", {"date": match_diary_date.group(1)})

    match_important_date = re.search(
        r"important date (.+) on (\d{4}-\d{2}-\d{2})(?: lead (\d+))?", lower
    )
    if match_important_date:
        return Intent(
            "important_date_add",
            {
                "name": match_important_date.group(1).strip(),
                "date": match_important_date.group(2),
                "lead_days": match_important_date.group(3) or "1",
            },
        )

    match_events_on = re.search(r"\bwhat(?:'s| is)\s+on\s+(\d{4}-\d{2}-\d{2})\b", lower)
    if match_events_on:
        return Intent("events_on_date", {"date": match_events_on.group(1)})

    match_reminder = re.search(r"remind me to (.+) at (\d{4}-\d{2}-\d{2} \d{2}:\d{2})", lower)
    if match_reminder:
        return Intent(
            "reminder_add",
            {
                "title": match_reminder.group(1).strip(),
                "at": _to_iso(match_reminder.group(2).strip()),
            },
        )

    match_alarm = re.search(r"set alarm (?:for|at) (\d{2}:\d{2})", lower)
    if match_alarm:
        return Intent("alarm_add", {"time": match_alarm.group(1)})

    if "list reminders" in lower:
        return Intent("reminder_list", {})

    if "list important dates" in lower:
        return Intent("important_date_list", {})

    return Intent("chat", {"text": t})


def _to_iso(time_str: str) -> str:
    dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M")
    return dt.isoformat(timespec="seconds")
