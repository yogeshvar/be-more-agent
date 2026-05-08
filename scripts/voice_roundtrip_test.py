#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
from contextlib import AsyncExitStack
from tempfile import NamedTemporaryFile

from beemboy.agent.orchestrator import AgentOrchestrator
from beemboy.config.settings import get_settings
from beemboy.llm.llama_server import LlamaServerBackend
from beemboy.mcp.bundle import MCPBundle
from beemboy.voice.audio import AudioConfig, MicrophoneLoop, write_pcm16_wav
from beemboy.voice.stt import WhisperCppSTT, WhisperSTTConfig
from beemboy.voice.tts import PiperTTS, PiperTTSConfig


class _NoopMCP:
    tools: list[object] = []

    async def invoke(self, openai_name: str, arguments_json: str) -> str:  # noqa: ARG002
        return ""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="One-shot voice roundtrip: mic -> whisper -> llama -> piper"
    )
    parser.add_argument("--seconds", type=float, default=5.0, help="Recording duration in seconds")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Recording sample rate in Hz")
    parser.add_argument("--input-device", default=None, help="Mic device index/name override")
    parser.add_argument("--skip-tts", action="store_true", help="Print reply but skip TTS playback")
    parser.add_argument(
        "--no-mcp",
        action="store_true",
        help="Disable MCP tools. By default MCP stays connected for the full session.",
    )
    return parser


async def _run_one_turn(
    args: argparse.Namespace,
    *,
    settings,
    orch: AgentOrchestrator,
    stt: WhisperCppSTT,
    tts: PiperTTS,
) -> int:
    print("Press Enter to start recording (or type 'q' to quit)...")
    if input().strip().lower() in {"q", "quit", "exit"}:
        return 99

    audio_cfg = AudioConfig(
        sample_rate=args.sample_rate,
        block_ms=settings.voice_chunk_ms,
        input_device=args.input_device if args.input_device else settings.voice_input_device,
    )
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
        print("Empty transcript; skipping turn.")
        return 0

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


async def _run(args: argparse.Namespace) -> int:
    settings = get_settings()
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
    llm = LlamaServerBackend(settings)
    async with AsyncExitStack() as stack:
        if args.no_mcp:
            mcp_client = _NoopMCP()
            print("MCP disabled for this session.")
        else:
            print("Connecting MCP servers once for this session...")
            mcp_client = await stack.enter_async_context(MCPBundle(settings.resolved_mcp_servers()))
            print("MCP ready.")
        orch = AgentOrchestrator(settings, llm, mcp_client)
        while True:
            rc = await _run_one_turn(args, settings=settings, orch=orch, stt=stt, tts=tts)
            if rc == 99:
                print("Bye.")
                return 0
            if rc != 0:
                print(f"Turn finished with warning code {rc}.")


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
