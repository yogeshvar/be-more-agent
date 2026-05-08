from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class VoicePhase(StrEnum):
    LISTENING_WAKE = "listening_wake"
    LISTENING_USER = "listening_user"
    THINKING = "thinking"
    SPEAKING = "speaking"
    FOLLOWUP_WINDOW = "followup_window"


@dataclass(slots=True)
class VoiceStateMachine:
    """Small state machine for wake/speech/reply/follow-up flow."""

    followup_enabled: bool = True
    phase: VoicePhase = VoicePhase.LISTENING_WAKE

    def on_wake_detected(self) -> VoicePhase:
        self.phase = VoicePhase.LISTENING_USER
        return self.phase

    def on_user_audio_captured(self, *, has_audio: bool) -> VoicePhase:
        if has_audio:
            self.phase = VoicePhase.THINKING
        else:
            self.phase = VoicePhase.LISTENING_WAKE
        return self.phase

    def on_agent_reply_ready(self) -> VoicePhase:
        self.phase = VoicePhase.SPEAKING
        return self.phase

    def on_tts_complete(self) -> VoicePhase:
        if self.followup_enabled:
            self.phase = VoicePhase.FOLLOWUP_WINDOW
        else:
            self.phase = VoicePhase.LISTENING_WAKE
        return self.phase

    def on_followup_window_end(self) -> VoicePhase:
        self.phase = VoicePhase.LISTENING_WAKE
        return self.phase

    def on_followup_speech_detected(self) -> VoicePhase:
        self.phase = VoicePhase.THINKING
        return self.phase
