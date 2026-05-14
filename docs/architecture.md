# Offline Assistant Architecture

## Service boundaries

- `beemboy.cli`: process lifecycle, text loop, runtime check command.
- `beemboy.orchestrator`: main coordinator from input to action/LLM response.
- `beemboy.intent`: deterministic intent and slot extraction for core commands.
- `beemboy.guardrails`: deterministic deny logic for latest/live/current-event asks.
- `beemboy.storage`: SQLite persistence for diary, important dates, reminders, alarms, and events.
- `beemboy.scheduler`: polling engine for due reminders and reminder-fired events.
- `beemboy.llm`: local `llama-cli` single-turn conversation wrapper.
- `beemboy.voice`: output speech (`say`) and text fallback capture.
- `beemboy.context_loader`: loads markdown context/policy files into prompt context.

## Request flow

1. User input arrives through text fallback (voice expansion point exists).
2. Guardrail check runs first.
3. If blocked, assistant returns deterministic offline-safe response.
4. If allowed, deterministic intent parsing checks for diary/reminder/alarm actions.
5. Structured actions are persisted through SQLite.
6. Free-form chat requests are sent to local `llama-cli` with context bundle.
7. Scheduler polls due reminders each loop and triggers spoken notifications.

## Offline guarantees

- No remote APIs are called by the app.
- All memory and schedules are local in SQLite.
- Policy and style context come from local markdown files.
