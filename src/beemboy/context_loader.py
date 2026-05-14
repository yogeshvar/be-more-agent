from __future__ import annotations

from pathlib import Path


DEFAULT_CONTEXT_FILES = [
    "assistant_identity.md",
    "capabilities.md",
    "guardrails.md",
    "response_templates.md",
    "user_preferences.md",
]


def ensure_default_context_files(context_dir: Path) -> None:
    context_dir.mkdir(parents=True, exist_ok=True)
    repo_context_dir = Path.cwd() / "context"
    defaults = {
        "assistant_identity.md": (
            "# Identity\n"
            "You are an offline personal assistant focused on reminders, alarms, and diary help.\n"
        ),
        "capabilities.md": (
            "# Capabilities\n"
            "- Diary write and retrieval by date/topic.\n"
            "- Reminder and alarm management.\n"
            "- Important date tracking and proactive reminders.\n"
        ),
        "guardrails.md": (
            "# Guardrails\n"
            "- No internet lookups.\n"
            "- Do not answer latest news/current-events requests with fabricated facts.\n"
        ),
        "response_templates.md": (
            "# Templates\n"
            "latest_news_denied: I work fully offline and cannot provide live or latest updates. "
            "I can help with your diary, reminders, alarms, and important dates.\n"
        ),
        "user_preferences.md": (
            "# User Preferences\n"
            "- Keep responses concise.\n"
            "- Confirm reminder/alarm changes clearly.\n"
        ),
    }
    for name, content in defaults.items():
        target = context_dir / name
        if target.exists():
            continue
        source_file = repo_context_dir / name
        if source_file.exists():
            target.write_text(source_file.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            target.write_text(content, encoding="utf-8")


def load_context_bundle(context_dir: Path) -> str:
    blocks: list[str] = []
    for name in DEFAULT_CONTEXT_FILES:
        fp = context_dir / name
        if fp.exists():
            blocks.append(f"## {name}\n{fp.read_text(encoding='utf-8').strip()}")
    return "\n\n".join(blocks)
