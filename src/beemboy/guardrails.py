from __future__ import annotations

from dataclasses import dataclass
import re


FORBIDDEN_PATTERNS = [
    r"\blatest\b",
    r"\bbreaking\s+news\b",
    r"\bnews\b",
    r"\bcurrent\s+events\b",
    r"\bwhat('?s| is)\s+happening\b",
    r"\bstock\s+price\b",
    r"\bweather\s+today\b",
    r"\blive\s+update(s)?\b",
]

DEFAULT_DENY_RESPONSE = (
    "I work fully offline and cannot provide live or latest updates. "
    "I can help with your diary, reminders, alarms, and important dates."
)


@dataclass(frozen=True)
class GuardrailResult:
    blocked: bool
    response: str | None = None
    reason: str | None = None


def check_guardrails(user_text: str, deny_response: str = DEFAULT_DENY_RESPONSE) -> GuardrailResult:
    normalized = user_text.strip().lower()
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, normalized):
            return GuardrailResult(
                blocked=True,
                response=deny_response,
                reason=f"matched:{pattern}",
            )
    return GuardrailResult(blocked=False)
