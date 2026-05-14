from pathlib import Path

from beemboy.context_loader import ensure_default_context_files
from beemboy.orchestrator import Orchestrator
from beemboy.scheduler import ReminderScheduler
from beemboy.storage import Store


class FakeLlm:
    def chat(self, _system_context: str, user_text: str) -> str:
        return f"fake:{user_text}"


def _build_app(tmp_path: Path) -> Orchestrator:
    context_dir = tmp_path / "context"
    ensure_default_context_files(context_dir)
    store = Store(tmp_path / "assistant.db")
    scheduler = ReminderScheduler(store, on_reminder=lambda _event: None)
    return Orchestrator(
        store=store,
        llm=FakeLlm(),  # type: ignore[arg-type]
        scheduler=scheduler,
        context_dir=str(context_dir),
    )


def test_guardrail_short_circuit(tmp_path: Path):
    app = _build_app(tmp_path)
    reply = app.handle_user_text("give me latest news")
    assert "offline" in reply.lower()


def test_diary_and_read(tmp_path: Path):
    app = _build_app(tmp_path)
    saved = app.handle_user_text("diary: met john and discussed launch")
    assert "Saved diary entry" in saved
    date = "2026-05-12"
    app.store.add_diary_entry("manual test entry", entry_date=date)
    read = app.handle_user_text(f"diary on {date}")
    assert "manual test entry" in read


def test_reminder_add(tmp_path: Path):
    app = _build_app(tmp_path)
    reply = app.handle_user_text("remind me to call mom at 2026-05-13 10:00")
    assert "Reminder #" in reply


def test_important_dates_and_events_lookup(tmp_path: Path):
    app = _build_app(tmp_path)
    save = app.handle_user_text("important date mom birthday on 2026-06-01 lead 2")
    assert "Saved important date" in save
    listed = app.handle_user_text("list important dates")
    assert "mom birthday" in listed
    on_day = app.handle_user_text("what is on 2026-06-01")
    assert "Important dates" in on_day
