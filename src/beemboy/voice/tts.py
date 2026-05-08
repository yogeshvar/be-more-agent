from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex
import subprocess
from tempfile import NamedTemporaryFile


@dataclass(slots=True)
class PiperTTSConfig:
    binary: str = "piper"
    model_path: str = "models/en_US-lessac-medium.onnx"
    playback_command: str | None = None


class PiperTTS:
    """Simple Piper CLI wrapper for offline TTS."""

    def __init__(self, config: PiperTTSConfig) -> None:
        self._config = config

    def _resolve_player(self) -> list[str]:
        if self._config.playback_command:
            return shlex.split(self._config.playback_command)
        return ["aplay"]

    def synthesize_to_wav(self, text: str, wav_path: str) -> None:
        cmd = [
            self._config.binary,
            "-m",
            self._config.model_path,
            "-f",
            wav_path,
        ]
        proc = subprocess.run(
            cmd,
            input=text,
            text=True,
            check=False,
            capture_output=True,
        )
        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            raise RuntimeError(f"piper failed ({proc.returncode}): {stderr}")

    def speak(self, text: str) -> None:
        cleaned = text.strip()
        if not cleaned:
            return
        with NamedTemporaryFile(prefix="beemboy_tts_", suffix=".wav", delete=False) as tmp:
            wav_path = Path(tmp.name)
        try:
            self.synthesize_to_wav(cleaned, str(wav_path))
            player = self._resolve_player()
            proc = subprocess.run([*player, str(wav_path)], check=False, capture_output=True, text=True)
            if proc.returncode != 0:
                stderr = (proc.stderr or "").strip()
                raise RuntimeError(f"audio playback failed ({proc.returncode}): {stderr}")
        finally:
            wav_path.unlink(missing_ok=True)
