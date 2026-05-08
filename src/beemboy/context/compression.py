from __future__ import annotations

import shlex
import subprocess
import sys


class ContextCompressor:
    """Small adapter for optional local Caveman compression."""

    _DEFAULT_THRESHOLD_TOKENS = 180

    def __init__(self, enabled: bool) -> None:
        self._enabled = enabled

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, len(text) // 4)

    def should_compress(self, text: str, threshold_tokens: int | None = None) -> bool:
        if not self._enabled:
            return False
        if len(text.strip()) < 120:
            return False
        threshold = threshold_tokens or self._DEFAULT_THRESHOLD_TOKENS
        return self._estimate_tokens(text) >= threshold

    @staticmethod
    def _contains_sensitive_markers(text: str) -> bool:
        lowered = text.lower()
        markers = ("password", "api_key", "secret", "private key", "ssh-rsa", "exact_quote")
        return any(marker in lowered for marker in markers)

    def compress_for_context(self, text: str) -> str:
        if not self.should_compress(text):
            return text
        if self._contains_sensitive_markers(text):
            return text

        commands: list[list[str]] = [
            [sys.executable, "caveman_compress_nlp.py", "compress", text],
            ["python3", "caveman_compress_nlp.py", "compress", text],
        ]
        for cmd in commands:
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
            except Exception:
                continue
            if proc.returncode != 0:
                continue
            out = proc.stdout.strip()
            if out:
                return out
        return text

    @staticmethod
    def compress_args_preview(command: list[str]) -> str:
        return shlex.join(command)
