from __future__ import annotations

import unittest

from beemboy.prompting.loader import PROMPT_FILES_IN_ORDER, PromptPackLoader


class PromptLoaderTests(unittest.TestCase):
    def test_prompt_loader_uses_stable_order(self) -> None:
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            prompts_dir = Path(tmp) / "prompts"
            prompts_dir.mkdir(parents=True)
            for name in PROMPT_FILES_IN_ORDER:
                (prompts_dir / name).write_text(name, encoding="utf-8")

            loader = PromptPackLoader(prompts_dir)
            sections = loader.load_sections()
            self.assertEqual(sections, PROMPT_FILES_IN_ORDER)
