# Offline Mac Voice Assistant

Fully offline local assistant for macOS using `llama-cli` + GGUF model.

Features:
- Diary entries with date-based retrieval
- Important date tracking
- Reminders and alarms
- Deterministic guardrails for latest-news/current-events requests
- Local markdown context and policy files
- Voice-first architecture with text fallback

## Quick start

1. Ensure `llama-cli` is installed and accessible on PATH.
2. Set model path:
   - `export BEEMBOY_MODEL_PATH=/absolute/path/to/llama-3.2-3b-instruct-q4_k_m.gguf`
3. (Optional voice-first) set local STT command that prints transcript to stdout:
   - `export BEEMBOY_STT_COMMAND='python local_stt_script.py'`
3. Install:
   - `python3 -m venv .venv`
   - `.venv/bin/python -m pip install -e '.[dev]'`
4. Run:
   - `.venv/bin/beemboy --runtime-check`
   - `.venv/bin/beemboy --text`
   - `.venv/bin/beemboy` (voice loop, uses `BEEMBOY_STT_COMMAND`)

## Text commands

- Save diary: `diary: met Rahul and discussed launch tasks`
- Read diary by date: `diary on 2026-05-12`
- Add reminder: `remind me to send report at 2026-05-13 09:00`
- Add alarm: `set alarm for 07:30`
- List reminders: `list reminders`

## Guardrail examples

- Input: `what is the latest news today?`
- Output: generic offline response (no fabricated current-events answer)

## Notes

- Runtime is designed to be offline after setup.
- If voice stack dependencies are unavailable, text mode still works.
