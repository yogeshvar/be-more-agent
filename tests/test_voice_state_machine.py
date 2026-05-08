from __future__ import annotations

import unittest

from beemboy.voice.state import VoicePhase, VoiceStateMachine


class VoiceStateMachineTests(unittest.TestCase):
    def test_happy_path_with_followup_enabled(self) -> None:
        sm = VoiceStateMachine(followup_enabled=True)
        self.assertEqual(sm.phase, VoicePhase.LISTENING_WAKE)

        self.assertEqual(sm.on_wake_detected(), VoicePhase.LISTENING_USER)
        self.assertEqual(sm.on_user_audio_captured(has_audio=True), VoicePhase.THINKING)
        self.assertEqual(sm.on_agent_reply_ready(), VoicePhase.SPEAKING)
        self.assertEqual(sm.on_tts_complete(), VoicePhase.FOLLOWUP_WINDOW)
        self.assertEqual(sm.on_followup_window_end(), VoicePhase.LISTENING_WAKE)

    def test_followup_speech_shortcuts_to_thinking(self) -> None:
        sm = VoiceStateMachine(followup_enabled=True)
        sm.on_wake_detected()
        sm.on_user_audio_captured(has_audio=True)
        sm.on_agent_reply_ready()
        sm.on_tts_complete()
        self.assertEqual(sm.phase, VoicePhase.FOLLOWUP_WINDOW)
        self.assertEqual(sm.on_followup_speech_detected(), VoicePhase.THINKING)

    def test_no_audio_returns_to_wake(self) -> None:
        sm = VoiceStateMachine(followup_enabled=True)
        sm.on_wake_detected()
        self.assertEqual(sm.on_user_audio_captured(has_audio=False), VoicePhase.LISTENING_WAKE)

    def test_followup_disabled_goes_back_to_wake(self) -> None:
        sm = VoiceStateMachine(followup_enabled=False)
        sm.on_wake_detected()
        sm.on_user_audio_captured(has_audio=True)
        sm.on_agent_reply_ready()
        self.assertEqual(sm.on_tts_complete(), VoicePhase.LISTENING_WAKE)
