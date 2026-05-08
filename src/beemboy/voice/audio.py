from __future__ import annotations

import contextlib
from dataclasses import dataclass
from time import monotonic
from typing import Any
import wave


def _rms_energy(pcm16_mono: bytes) -> float:
    if not pcm16_mono:
        return 0.0
    sample_count = len(pcm16_mono) // 2
    if sample_count == 0:
        return 0.0
    total = 0.0
    for i in range(0, len(pcm16_mono), 2):
        sample = int.from_bytes(pcm16_mono[i : i + 2], byteorder="little", signed=True)
        norm = sample / 32768.0
        total += norm * norm
    return (total / sample_count) ** 0.5


@dataclass(slots=True)
class AudioConfig:
    sample_rate: int = 16000
    block_ms: int = 80
    input_device: str | None = None

    @property
    def frames_per_block(self) -> int:
        return max(80, int(self.sample_rate * self.block_ms / 1000))


class MicrophoneLoop:
    """Simple blocking microphone reader for 16-bit mono PCM."""

    def __init__(self, config: AudioConfig) -> None:
        self._config = config
        self._stream: Any | None = None
        self._sd: Any | None = None

    def _describe_input_devices(self, sd: Any) -> str:
        try:
            devices = sd.query_devices()
        except Exception:
            return "Could not query audio devices via PortAudio."

        input_rows: list[str] = []
        for index, info in enumerate(devices):
            try:
                max_input_channels = int(info.get("max_input_channels", 0))
                name = str(info.get("name", f"device-{index}")).strip() or f"device-{index}"
            except Exception:
                continue
            if max_input_channels > 0:
                input_rows.append(f"{index}: {name}")

        if not input_rows:
            return "No input devices were detected. Connect a microphone and try again."

        preview = ", ".join(input_rows[:8])
        if len(input_rows) > 8:
            preview += ", ..."
        return (
            "Available input devices: "
            f"{preview}. Set VOICE_INPUT_DEVICE to one of the device indexes above."
        )

    def open(self) -> None:
        try:
            import sounddevice as sd  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "sounddevice is required for voice mode. Install with: pip install sounddevice"
            ) from exc
        self._sd = sd
        device = self._config.input_device or None
        try:
            self._stream = sd.RawInputStream(
                samplerate=self._config.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=self._config.frames_per_block,
                device=device,
            )
        except Exception as exc:
            device_text = f"configured device={device!r}" if device is not None else "default input device"
            details = self._describe_input_devices(sd)
            raise RuntimeError(f"Unable to open microphone ({device_text}). {details}") from exc
        self._stream.start()

    def close(self) -> None:
        if self._stream is not None:
            with contextlib.suppress(Exception):
                self._stream.stop()
            with contextlib.suppress(Exception):
                self._stream.close()
        self._stream = None
        self._sd = None

    def __enter__(self) -> MicrophoneLoop:
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def read_chunk(self) -> bytes:
        if self._stream is None:
            raise RuntimeError("MicrophoneLoop is not open")
        data, overflowed = self._stream.read(self._config.frames_per_block)
        if overflowed:
            # Ignore overflows in favor of keeping the loop responsive.
            pass
        return bytes(data)

    def record_until_silence(
        self,
        *,
        max_duration_s: float,
        silence_duration_s: float,
        energy_threshold: float,
        wait_for_voice_timeout_s: float,
    ) -> bytes:
        started = monotonic()
        voice_started = False
        last_voice = started
        audio_parts: list[bytes] = []
        while True:
            now = monotonic()
            elapsed = now - started
            if elapsed >= max_duration_s:
                break
            if (not voice_started) and elapsed >= wait_for_voice_timeout_s:
                break

            chunk = self.read_chunk()
            energy = _rms_energy(chunk)
            if energy >= energy_threshold:
                voice_started = True
                last_voice = now
                audio_parts.append(chunk)
                continue

            if voice_started:
                audio_parts.append(chunk)
                if (now - last_voice) >= silence_duration_s:
                    break

        return b"".join(audio_parts)


def write_pcm16_wav(path: str, pcm16_audio: bytes, *, sample_rate: int) -> None:
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16_audio)
