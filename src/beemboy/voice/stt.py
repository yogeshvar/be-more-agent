from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess


@dataclass(slots=True)
class WhisperSTTConfig:
    binary: str = "whisper-cli"
    model_path: str = "models/ggml-base.en.bin"


class WhisperCppSTT:
    """Wrapper around whisper.cpp CLI for offline speech-to-text."""

    def __init__(self, config: WhisperSTTConfig) -> None:
        self._config = config

    def transcribe_wav(self, wav_path: str) -> str:
        input_path = Path(wav_path)
        output_prefix = str(input_path.with_suffix(""))
        cmd = [
            self._config.binary,
            "-m",
            self._config.model_path,
            "-f",
            str(input_path),
            "-otxt",
            "-of",
            output_prefix,
            "--no-timestamps",
        ]
        try:
            proc = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"whisper binary '{self._config.binary}' was not found on PATH. "
                "Install/compile whisper.cpp and either add `whisper-cli` to PATH "
                "or set VOICE_WHISPER_BINARY to its absolute path."
            ) from exc
        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            raise RuntimeError(f"whisper-cli failed ({proc.returncode}): {stderr}")
        txt_path = Path(output_prefix + ".txt")
        if txt_path.exists():
            return txt_path.read_text(encoding="utf-8").strip()
        return (proc.stdout or "").strip()
