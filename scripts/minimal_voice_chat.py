#!/usr/bin/env python3
from __future__ import annotations

import argparse

from openai import OpenAI

from beemboy.config.settings import get_settings
from beemboy.voice.tts import PiperTTS, PiperTTSConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Minimal terminal chat: llama-server text response + Piper voice output."
    )
    parser.add_argument(
        "--system",
        default="You are a concise helpful assistant.",
        help="System prompt sent with each turn.",
    )
    parser.add_argument(
        "--no-voice",
        action="store_true",
        help="Print replies without speaking them.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    settings = get_settings()

    client = OpenAI(
        base_url=settings.llama_base_url.rstrip("/"),
        api_key=settings.llama_api_key,
        timeout=settings.request_timeout_s,
    )
    tts = PiperTTS(
        PiperTTSConfig(
            binary=settings.voice_piper_binary,
            model_path=settings.voice_piper_model_path,
            playback_command=settings.voice_playback_command,
        )
    )

    print("Minimal voice chat ready. Type your message and press Enter.")
    print("Type 'exit' or 'quit' to stop.")
    while True:
        user_text = input("\nYou: ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            print("Bye.")
            return 0

        completion = client.chat.completions.create(
            model=settings.llama_model,
            temperature=settings.temperature,
            messages=[
                {"role": "system", "content": args.system},
                {"role": "user", "content": user_text},
            ],
        )
        reply = (completion.choices[0].message.content or "").strip()
        if not reply:
            print("Assistant: [empty response]")
            continue

        print(f"Assistant: {reply}")
        if args.no_voice:
            continue
        try:
            tts.speak(reply)
        except RuntimeError as exc:
            print(f"[voice disabled] {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
