from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Settings:
    model_path: str
    llama_cli: str
    app_home: Path
    db_path: Path
    context_dir: Path
    timezone: str
    stt_command: str
    tts_enabled: bool


def load_settings() -> Settings:
    app_home = Path(os.environ.get("BEEMBOY_HOME", "~/.beemboy")).expanduser()
    db_path = app_home / "assistant.db"
    context_dir = app_home / "context"
    return Settings(
        model_path=os.environ.get("BEEMBOY_MODEL_PATH", "").strip(),
        llama_cli=os.environ.get("BEEMBOY_LLAMA_CLI", "llama-cli"),
        app_home=app_home,
        db_path=db_path,
        context_dir=context_dir,
        timezone=os.environ.get("BEEMBOY_TIMEZONE", "local"),
        stt_command=os.environ.get("BEEMBOY_STT_COMMAND", "").strip(),
        tts_enabled=os.environ.get("BEEMBOY_TTS_ENABLED", "1").strip() != "0",
    )
