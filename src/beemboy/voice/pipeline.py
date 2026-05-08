from __future__ import annotations

import asyncio
from dataclasses import dataclass
from time import perf_counter
from typing import Callable
from tempfile import NamedTemporaryFile

import structlog

from beemboy.agent.orchestrator import AgentOrchestrator
from beemboy.agent.telemetry import TurnTelemetry, VoiceTurnTelemetry
from beemboy.config.settings import Settings
from beemboy.voice.audio import AudioConfig, MicrophoneLoop, write_pcm16_wav
from beemboy.voice.state import VoicePhase, VoiceStateMachine
from beemboy.voice.stt import WhisperCppSTT, WhisperSTTConfig
from beemboy.voice.tts import PiperTTS, PiperTTSConfig
from beemboy.voice.wake import OnnxWakeWordDetector, WakeDetectorConfig

log = structlog.get_logger(__name__)


@dataclass(slots=True)
class VoiceLoopEvents:
    on_status: Callable[[str], None] | None = None
    on_telemetry: Callable[[int, TurnTelemetry, VoiceTurnTelemetry], None] | None = None


class VoiceAssistantLoop:
    """Offline voice stack: wake -> stt -> orchestrator -> tts -> follow-up."""

    def __init__(self, settings: Settings, orchestrator: AgentOrchestrator, events: VoiceLoopEvents | None = None) -> None:
        self._settings = settings
        self._orchestrator = orchestrator
        self._events = events or VoiceLoopEvents()
        self._state = VoiceStateMachine(followup_enabled=settings.voice_followup_seconds > 0.0)
        self._wake = OnnxWakeWordDetector(
            WakeDetectorConfig(
                model_path=settings.voice_wake_model_path,
                threshold=settings.voice_wake_threshold,
                trigger_level=settings.voice_wake_trigger_level,
                refractory_seconds=settings.voice_wake_refractory_s,
            )
        )
        self._stt = WhisperCppSTT(
            WhisperSTTConfig(
                binary=settings.voice_whisper_binary,
                model_path=settings.voice_whisper_model_path,
            )
        )
        self._tts = PiperTTS(
            PiperTTSConfig(
                binary=settings.voice_piper_binary,
                model_path=settings.voice_piper_model_path,
                playback_command=settings.voice_playback_command,
            )
        )
        self._audio_config = AudioConfig(
            sample_rate=settings.voice_sample_rate,
            block_ms=settings.voice_chunk_ms,
            input_device=settings.voice_input_device,
        )
        self._history: list[dict] = []

    def _emit_status(self, text: str) -> None:
        if self._events.on_status:
            self._events.on_status(text)

    async def _audio_to_text(self, pcm16_audio: bytes) -> tuple[str, float]:
        with NamedTemporaryFile(prefix="beemboy_user_", suffix=".wav", delete=True) as tmp:
            write_pcm16_wav(tmp.name, pcm16_audio, sample_rate=self._settings.voice_sample_rate)
            started = perf_counter()
            text = await asyncio.to_thread(self._stt.transcribe_wav, tmp.name)
            latency_ms = (perf_counter() - started) * 1000
            return text.strip(), latency_ms

    async def _capture_user_audio(
        self,
        mic: MicrophoneLoop,
        *,
        wake_already_triggered: bool,
    ) -> tuple[bytes, VoicePhase]:
        self._state.on_wake_detected()
        self._emit_status("Listening...")
        wait_timeout = (
            self._settings.voice_followup_seconds
            if not wake_already_triggered
            else self._settings.voice_post_wake_timeout_s
        )
        audio = await asyncio.to_thread(
            mic.record_until_silence,
            max_duration_s=self._settings.voice_max_utterance_seconds,
            silence_duration_s=self._settings.voice_silence_seconds,
            energy_threshold=self._settings.voice_energy_threshold,
            wait_for_voice_timeout_s=wait_timeout,
        )
        phase = self._state.on_user_audio_captured(has_audio=bool(audio))
        return audio, phase

    async def run(self) -> None:
        self._emit_status(f"Voice loop ready. Say '{self._settings.voice_wake_phrase}'. Ctrl+C to exit.")
        turn_index = 0
        with MicrophoneLoop(self._audio_config) as mic:
            while True:
                # Phase 1: listen for wake word
                self._state.phase = VoicePhase.LISTENING_WAKE
                while self._state.phase == VoicePhase.LISTENING_WAKE:
                    chunk = await asyncio.to_thread(mic.read_chunk)
                    if self._wake.process_chunk(chunk):
                        self._emit_status("Wake word detected.")
                        break

                user_audio, phase_after_capture = await self._capture_user_audio(
                    mic,
                    wake_already_triggered=True,
                )
                if phase_after_capture != VoicePhase.THINKING:
                    continue

                while True:
                    turn_index += 1
                    voice_telemetry = VoiceTurnTelemetry()
                    stt_started = perf_counter()
                    transcript, stt_latency_ms = await self._audio_to_text(user_audio)
                    voice_telemetry.stt_latency_ms = stt_latency_ms
                    voice_telemetry.user_audio_ms = ((perf_counter() - stt_started) * 1000) - stt_latency_ms
                    if not transcript:
                        self._emit_status("Didn't catch that.")
                        break

                    self._emit_status(f"You: {transcript}")
                    turn_telemetry: TurnTelemetry | None = None

                    def _capture_turn_telemetry(value: TurnTelemetry) -> None:
                        nonlocal turn_telemetry
                        turn_telemetry = value

                    self._state.phase = VoicePhase.THINKING
                    self._history, reply = await self._orchestrator.run_turn(
                        self._history,
                        transcript,
                        on_telemetry=_capture_turn_telemetry,
                    )
                    self._state.on_agent_reply_ready()
                    self._emit_status(f"{self._settings.assistant_name}: {reply}")
                    tts_start = perf_counter()
                    await asyncio.to_thread(self._tts.speak, reply)
                    voice_telemetry.tts_latency_ms = (perf_counter() - tts_start) * 1000
                    self._state.on_tts_complete()
                    if turn_telemetry and self._events.on_telemetry:
                        self._events.on_telemetry(turn_index, turn_telemetry, voice_telemetry)

                    if self._state.phase != VoicePhase.FOLLOWUP_WINDOW:
                        break
                    self._emit_status("Follow-up window open...")
                    followup_audio = await asyncio.to_thread(
                        mic.record_until_silence,
                        max_duration_s=self._settings.voice_max_utterance_seconds,
                        silence_duration_s=self._settings.voice_silence_seconds,
                        energy_threshold=self._settings.voice_energy_threshold,
                        wait_for_voice_timeout_s=self._settings.voice_followup_seconds,
                    )
                    if not followup_audio:
                        self._state.on_followup_window_end()
                        break
                    self._state.on_followup_speech_detected()
                    user_audio = followup_audio
                    self._emit_status("Follow-up heard.")
