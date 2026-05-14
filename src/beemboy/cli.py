from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

from .config import load_settings
from .context_loader import ensure_default_context_files
from .llm import LlamaClient
from .orchestrator import Orchestrator
from .runtime_check import check_runtime
from .scheduler import ReminderScheduler
from .storage import Store
from .voice import capture_from_stt_command, capture_text_fallback, speak


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline Mac voice assistant")
    parser.add_argument("--text", action="store_true", help="run in text interactive mode")
    parser.add_argument("--once", type=str, default="", help="single-turn text query")
    parser.add_argument(
        "--runtime-check", action="store_true", help="show llama runtime + model status and exit"
    )
    args = parser.parse_args()

    settings = load_settings()
    settings.app_home.mkdir(parents=True, exist_ok=True)
    ensure_default_context_files(settings.context_dir)

    if args.runtime_check:
        return _print_runtime_status(settings)

    if not settings.model_path:
        print("BEEMBOY_MODEL_PATH is required.")
        return 2

    store = Store(settings.db_path)
    llm = LlamaClient(settings.llama_cli, settings.model_path)
    scheduler = ReminderScheduler(store, on_reminder=lambda _event: None)
    app = Orchestrator(
        store=store,
        llm=llm,
        scheduler=scheduler,
        context_dir=str(settings.context_dir),
    )

    if args.once:
        reply = app.handle_user_text(args.once)
        print(reply)
        return 0

    mode = "text" if args.text else "voice"
    print(f"Offline assistant ready in {mode} mode. Type 'exit' to quit.")
    while True:
        for fired in app.tick_scheduler():
            print(f"[reminder] {fired}")
            speak(fired, enabled=settings.tts_enabled)

        try:
            if args.text:
                user_text = capture_text_fallback()
            else:
                user_text = capture_from_stt_command(settings.stt_command)
                if user_text:
                    print(f"Heard> {user_text}")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return 0

        if not user_text:
            time.sleep(0.05)
            continue
        if user_text.lower() in {"exit", "quit"}:
            print("Goodbye.")
            return 0

        reply = app.handle_user_text(user_text)
        print(f"Assistant> {reply}")
        speak(reply, enabled=settings.tts_enabled)


def _print_runtime_status(settings) -> int:
    try:
        status = check_runtime(settings)
    except RuntimeError as exc:
        print(str(exc))
        return 1

    print(f"llama-cli: {status.llama_cli_path}")
    print(f"version: {status.llama_version}")
    print(f"conversation_flag: {status.has_conversation_flag}")
    print(f"model_exists: {status.model_exists}")
    print(f"model_probe_ok: {status.model_probe_ok}")
    if settings.model_path:
        print(f"model_path: {Path(settings.model_path).expanduser()}")
    else:
        print("model_path: <unset>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
