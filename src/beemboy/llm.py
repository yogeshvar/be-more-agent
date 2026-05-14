from __future__ import annotations

from dataclasses import dataclass
import subprocess


@dataclass(frozen=True)
class LlamaClient:
    llama_cli: str
    model_path: str
    n_threads: int = 6
    n_ctx: int = 4096
    n_gpu_layers: int = 99

    def chat(self, system_context: str, user_text: str) -> str:
        prompt = (
            "System context:\n"
            f"{system_context}\n\n"
            "User message:\n"
            f"{user_text}\n\n"
            "Respond with concise helpful text."
        )
        proc = subprocess.run(
            [
                self.llama_cli,
                "-m",
                self.model_path,
                "-cnv",
                "--single-turn",
                "-t",
                str(self.n_threads),
                "-c",
                str(self.n_ctx),
                "-ngl",
                str(self.n_gpu_layers),
                "-p",
                prompt,
            ],
            text=True,
            capture_output=True,
        )
        if proc.returncode != 0:
            return f"Local model error: {proc.stderr.strip() or 'unknown llama-cli failure'}"

        text = (proc.stdout or "").strip()
        if not text:
            return "I could not generate a response right now."
        return text.splitlines()[-1].strip()
