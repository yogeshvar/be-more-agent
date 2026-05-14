# Offline Acceptance Test Matrix

## Runtime baseline

- `beemboy --runtime-check` reports:
  - llama-cli path,
  - version,
  - conversation flag support,
  - model file existence.

## Diary behavior

- Input: `diary: today I focused on project planning`
  - Expected: entry saved confirmation.
- Input: `diary on YYYY-MM-DD`
  - Expected: list of entries for date or explicit "no entries."

## Reminder behavior

- Input: `remind me to submit report at 2026-05-13 09:00`
  - Expected: reminder id returned and stored.
- Input after due time:
  - Expected: reminder event emitted, logged, marked done.

## Alarm behavior

- Input: `set alarm for 07:30`
  - Expected: daily alarm stored and confirmed.

## Guardrail behavior

- Input: "what is the latest news?"
  - Expected: fixed generic offline-safe denial response.
- Input: "weather today?"
  - Expected: same denial path.

## Persistence behavior

- Restart app after creating diary/reminders.
  - Expected: data remains available.

## Offline behavior

- Disconnect internet and run all flows.
  - Expected: all assistant core features still work.
