from __future__ import annotations

from pathlib import Path


PROMPT_FILES_IN_ORDER = [
    "core_persona.md",
    "response_style.md",
    "memory_usage.md",
    "tool_policy.md",
    "vision_policy.md",
]


class PromptPackLoader:
    """Loads prompt markdown files in deterministic order."""

    def __init__(self, prompts_dir: str | Path | None = None) -> None:
        if prompts_dir is None:
            root = Path(__file__).resolve().parents[3]
            self._prompts_dir = root / "prompts"
        else:
            self._prompts_dir = Path(prompts_dir)

    @property
    def prompts_dir(self) -> Path:
        return self._prompts_dir

    def load_sections(self) -> list[str]:
        blocks: list[str] = []
        for filename in PROMPT_FILES_IN_ORDER:
            path = self._prompts_dir / filename
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8").strip()
            if text:
                blocks.append(text)
        return blocks

    def compose(self, extra_blocks: list[str] | None = None) -> str:
        sections = self.load_sections()
        if extra_blocks:
            sections.extend(block for block in extra_blocks if block and block.strip())
        return "\n\n".join(sections).strip()
