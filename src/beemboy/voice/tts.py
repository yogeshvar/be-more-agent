from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex
import shutil
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
        for cmd in ("pw-play", "paplay", "aplay"):
            if shutil.which(cmd):
                return [cmd]
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
            try:
                self.synthesize_to_wav(cleaned, str(wav_path))
            except FileNotFoundError as exc:
                raise RuntimeError(
                    f"piper binary '{self._config.binary}' was not found on PATH. "
                    "Install piper and either add it to PATH or set VOICE_PIPER_BINARY "
                    "to its absolute path."
                ) from exc
            player = self._resolve_player()
            try:
                proc = subprocess.run([*player, str(wav_path)], check=False, capture_output=True, text=True)
            except FileNotFoundError as exc:
                raise RuntimeError(
                    f"Audio playback command '{' '.join(player)}' was not found. "
                    "Install a player (for example `aplay`) or set VOICE_PLAYBACK_COMMAND."
                ) from exc
            if proc.returncode != 0 and (not self._config.playback_command) and player and player[0] != "aplay":
                # PipeWire/Pulse fallback to ALSA can help on systems with mixed audio stacks.
                if shutil.which("aplay"):
                    proc = subprocess.run(["aplay", str(wav_path)], check=False, capture_output=True, text=True)
            if proc.returncode != 0:
                stderr = (proc.stderr or "").strip()
                raise RuntimeError(f"audio playback failed ({proc.returncode}): {stderr}")
        finally:
            wav_path.unlink(missing_ok=True)
