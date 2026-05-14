from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess

from .config import Settings


@dataclass(frozen=True)
class RuntimeStatus:
    llama_cli_path: str
    llama_version: str
    has_conversation_flag: bool
    model_exists: bool
    model_probe_ok: bool


def check_runtime(settings: Settings) -> RuntimeStatus:
    llama_path = shutil.which(settings.llama_cli)
    if not llama_path:
        raise RuntimeError(f"Cannot find llama-cli binary: {settings.llama_cli}")

    version_proc = subprocess.run(
        [llama_path, "--version"],
        check=True,
        text=True,
        capture_output=True,
    )
    help_proc = subprocess.run(
        [llama_path, "--help"],
        check=True,
        text=True,
        capture_output=True,
    )
    help_text = help_proc.stdout + "\n" + help_proc.stderr
    has_conversation = "--conversation" in help_text or "-cnv" in help_text
    model_exists = bool(settings.model_path) and Path(settings.model_path).exists()
    model_probe_ok = False
    if model_exists:
        try:
            probe = subprocess.run(
                [
                    llama_path,
                    "-m",
                    settings.model_path,
                    "--single-turn",
                    "-cnv",
                    "-n",
                    "8",
                    "-p",
                    "Say ok.",
                ],
                text=True,
                capture_output=True,
                timeout=30,
            )
            model_probe_ok = probe.returncode == 0
        except subprocess.SubprocessError:
            model_probe_ok = False

    return RuntimeStatus(
        llama_cli_path=llama_path,
        llama_version=(version_proc.stdout or version_proc.stderr).strip(),
        has_conversation_flag=has_conversation,
        model_exists=model_exists,
        model_probe_ok=model_probe_ok,
    )
