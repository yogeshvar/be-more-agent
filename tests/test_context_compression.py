from __future__ import annotations

from unittest.mock import patch
import unittest

from beemboy.context.compression import ContextCompressor


class ContextCompressionTests(unittest.TestCase):
    def test_should_compress_threshold(self) -> None:
        compressor = ContextCompressor(enabled=True)
        text = "token " * 300
        self.assertTrue(compressor.should_compress(text))
        self.assertFalse(compressor.should_compress("short text"))

    def test_compressor_falls_back_on_errors(self) -> None:
        compressor = ContextCompressor(enabled=True)
        text = "very long context " * 120

        with patch("beemboy.context.compression.subprocess.run", side_effect=RuntimeError("boom")):
            self.assertEqual(compressor.compress_for_context(text), text)
