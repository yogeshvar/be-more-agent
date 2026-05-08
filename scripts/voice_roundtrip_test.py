#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
from tempfile import NamedTemporaryFile

from beemboy.agent.orchestrator import AgentOrchestrator
from beemboy.config.settings import get_settings
from beemboy.llm.llama_server import LlamaServerBackend
from beemboy.mcp.bundle import MCPBundle
from beemboy.voice.audio import AudioConfig, MicrophoneLoop, write_pcm16_wav
from beemboy.voice.stt import WhisperCppSTT, WhisperSTTConfig
from beemboy.voice.tts import PiperTTS, PiperTTSConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="One-shot voice roundtrip: mic -> whisper -> llama -> piper"
    )
    parser.add_argument("--seconds", type=float, default=5.0, help="Recording duration in seconds")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Recording sample rate in Hz")
    parser.add_argument("--input-device", default=None, help="Mic device index/name override")
    parser.add_argument("--skip-tts", action="store_true", help="Print reply but skip TTS playback")
    return parser


async def _run(args: argparse.Namespace) -> int:
    settings = get_settings()
    audio_cfg = AudioConfig(
        sample_rate=args.sample_rate,
        block_ms=settings.voice_chunk_ms,
        input_device=args.input_device if args.input_device else settings.voice_input_device,
    )
    stt = WhisperCppSTT(
        WhisperSTTConfig(
            binary=settings.voice_whisper_binary,
            model_path=settings.voice_whisper_model_path,
        )
    )
    tts = PiperTTS(
        PiperTTSConfig(
            binary=settings.voice_piper_binary,
            model_path=settings.voice_piper_model_path,
            playback_command=settings.voice_playback_command,
        )
    )

    print("Press Enter to start recording...")
    input()

    pcm_parts: list[bytes] = []
    target_frames = max(1, int(args.seconds * args.sample_rate))
    captured_frames = 0
    print(f"Recording for {args.seconds:.1f}s...")
    with MicrophoneLoop(audio_cfg) as mic:
        while captured_frames < target_frames:
            chunk = await asyncio.to_thread(mic.read_chunk)
            pcm_parts.append(chunk)
            captured_frames += len(chunk) // 2
    pcm16_audio = b"".join(pcm_parts)

    if not pcm16_audio:
        print("No microphone audio captured.")
        return 1

    with NamedTemporaryFile(prefix="beemboy_roundtrip_", suffix=".wav", delete=True) as tmp:
        write_pcm16_wav(tmp.name, pcm16_audio, sample_rate=args.sample_rate)
        transcript = await asyncio.to_thread(stt.transcribe_wav, tmp.name)

    transcript = transcript.strip()
    print(f"\nYou said: {transcript or '[empty]'}")
    if not transcript:
        print("Empty transcript; stopping.")
        return 2

    servers = settings.resolved_mcp_servers()
    async with MCPBundle(servers) as mcp:
        llm = LlamaServerBackend(settings)
        orch = AgentOrchestrator(settings, llm, mcp)
        _history, reply = await orch.run_turn([], transcript)

    reply = (reply or "").strip()
    print(f"\nAssistant: {reply or '[empty]'}")
    if not reply:
        return 3

    if not args.skip_tts:
        print("\nSpeaking response...")
        await asyncio.to_thread(tts.speak, reply)
        print("Done.")
    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
