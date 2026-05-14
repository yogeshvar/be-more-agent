from __future__ import annotations

import subprocess


def speak(text: str, enabled: bool = True) -> None:
    if not enabled:
        return
    # macOS built-in TTS. Fails silently in headless/test contexts.
    subprocess.run(["say", text], check=False)


def capture_text_fallback() -> str:
    return input("You> ").strip()


def capture_from_stt_command(command: str) -> str:
    if not command:
        return ""
    proc = subprocess.run(command, shell=True, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        return ""
    return (proc.stdout or "").strip()
