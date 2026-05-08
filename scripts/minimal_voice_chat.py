#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path

from openai import OpenAI

from beemboy.config.settings import get_settings
from beemboy.vision.detector import FaceDetector
from beemboy.vision.embedder import FaceEmbedder
from beemboy.vision.registry import FaceRegistry
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
    parser.add_argument(
        "--face-image",
        required=False,
        default=None,
        help="Path to an image containing your face for identity check/enrollment.",
    )
    return parser


def _say(text: str, *, tts: PiperTTS, no_voice: bool) -> None:
    print(f"Assistant: {text}")
    if no_voice:
        return
    try:
        tts.speak(text)
    except RuntimeError as exc:
        print(f"[voice disabled] {exc}")


def _opening_question(name: str) -> str:
    prompts = [
        f"Hey {name}, what's one cool thing you want to do today?",
        f"Good to see you {name}. What should we build right now?",
        f"{name}, what's one question you want me to answer first?",
        f"Welcome back {name}. What's on your mind?",
    ]
    return random.choice(prompts)


def _bootstrap_identity(face_image_path: str, *, settings, tts: PiperTTS, no_voice: bool) -> str:
    image_path = Path(face_image_path).expanduser()
    if not image_path.is_file():
        raise FileNotFoundError(f"Face image not found: {image_path}")

    image_bytes = image_path.read_bytes()
    detector = FaceDetector(
        backend=settings.camera_detector_backend,
        min_face_size_px=settings.camera_min_face_size_px,
    )
    embedder = FaceEmbedder()
    registry = FaceRegistry(settings.camera_identity_store_path)

    faces = detector.detect(image_bytes)
    if not faces:
        raise RuntimeError(
            "No face detected. Make sure OpenCV is installed and the image has a clear frontal face."
        )
    embedding = embedder.embed(faces[0].crop_bytes)
    if not embedding:
        raise RuntimeError("Failed to generate a face embedding from the detected face.")

    match = registry.match(embedding, threshold=settings.camera_match_threshold)
    if match.matched and match.name:
        opener = _opening_question(match.name)
        _say(opener, tts=tts, no_voice=no_voice)
        return match.name

    _say("I don't recognize you yet. Who are you?", tts=tts, no_voice=no_voice)
    name = input("You (name): ").strip() or "Friend"
    enrolled = registry.enroll(name=name, embedding=embedding)
    _say(f"Nice to meet you, {enrolled.name}.", tts=tts, no_voice=no_voice)
    return enrolled.name


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

    person_name = "Friend"
    if args.face_image:
        try:
            person_name = _bootstrap_identity(
                args.face_image,
                settings=settings,
                tts=tts,
                no_voice=args.no_voice,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(f"Identity setup error: {exc}")
            return 1
        print(f"Identity ready for {person_name}.")
    else:
        print("No --face-image provided; starting chat without identity recognition.")

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
                {"role": "system", "content": f"{args.system} The user's name is {person_name}."},
                {"role": "user", "content": user_text},
            ],
        )
        reply = (completion.choices[0].message.content or "").strip()
        if not reply:
            print("Assistant: [empty response]")
            continue

        _say(reply, tts=tts, no_voice=args.no_voice)


if __name__ == "__main__":
    raise SystemExit(main())
