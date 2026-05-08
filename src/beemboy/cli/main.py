from __future__ import annotations

import asyncio
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from typing import Any

import structlog
import typer

from beemboy.agent.orchestrator import AgentOrchestrator
from beemboy.agent.telemetry import TurnTelemetry, VoiceTurnTelemetry
from beemboy.config.settings import Settings, get_settings
from beemboy.llm.llama_server import LlamaServerBackend
from beemboy.mcp.bundle import MCPBundle
from beemboy.ui.controller import UIController

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
)

app = typer.Typer(help="Beemboy: local llama-server + MCP (Brave Search) assistant.", no_args_is_help=True)


def _configure_logging(verbose: bool) -> None:
    import logging

    level = logging.DEBUG if verbose else logging.INFO
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
    )


def _emit_turn_telemetry(turn_index: int, telemetry: TurnTelemetry) -> None:
    payload: dict[str, Any] = {
        "type": "turn_telemetry",
        "turn": turn_index,
        "metrics": telemetry.to_debug_dict(),
    }
    typer.secho(f"[telemetry] {json.dumps(payload, ensure_ascii=False)}", dim=True, err=True)


def _emit_voice_telemetry(turn_index: int, telemetry: TurnTelemetry, voice: VoiceTurnTelemetry) -> None:
    payload: dict[str, Any] = {
        "type": "voice_turn_telemetry",
        "turn": turn_index,
        "metrics": telemetry.to_debug_dict(),
        "voice": voice.to_debug_dict(),
    }
    typer.secho(f"[telemetry] {json.dumps(payload, ensure_ascii=False)}", dim=True, err=True)


def _parse_embedding_arg(raw: str) -> list[float]:
    chunks = [part.strip() for part in raw.replace(" ", ",").split(",") if part.strip()]
    if not chunks:
        raise ValueError("No embedding values provided")
    return [float(item) for item in chunks]


class _NoopLLM:
    async def chat(self, messages, tools=None, tool_choice=None):  # noqa: ANN001, ARG002
        msg = SimpleNamespace(content="", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    async def stream_complete(self, messages, tools=None, tool_choice=None, on_text_delta=None):  # noqa: ANN001, ARG002
        return {"role": "assistant", "content": "", "tool_calls": None}


class _NoopMCP:
    tools: list[Any] = []

    async def invoke(self, openai_name: str, arguments_json: str) -> str:  # noqa: ARG002
        return ""


async def _chat_loop(settings: Settings, *, telemetry_enabled: bool = False) -> None:
    try:
        servers = settings.resolved_mcp_servers()
    except (json.JSONDecodeError, ValueError) as e:
        typer.secho(f"Invalid MCP_SERVERS or MCP config: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from e
    log = structlog.get_logger("cli")
    default_mcp_expected = (
        settings.default_mcp_enabled
        and not (settings.brave_api_key or "").strip()
        and not (settings.mcp_servers or "").strip()
        and not (settings.mcp_proxy_base_url or "").strip()
    )
    if default_mcp_expected and shutil.which("uvx") is None:
        typer.secho(
            "Default MCP servers are enabled, but 'uvx' is not installed. "
            "Install uv (https://docs.astral.sh/uv/getting-started/installation/) "
            "or set DEFAULT_MCP_ENABLED=false.",
            fg=typer.colors.YELLOW,
            err=True,
        )
    if not servers:
        typer.secho(
            "No MCP servers: set BRAVE_API_KEY and/or MCP_SERVERS. Chat will run without tools.",
            fg=typer.colors.YELLOW,
            err=True,
        )
    async with MCPBundle(servers) as mcp:
        llm = LlamaServerBackend(settings)
        orch = AgentOrchestrator(settings, llm, mcp)
        history: list[dict] = []
        tool_names = ", ".join(t.openai_name for t in mcp.tools[:8])
        if len(mcp.tools) > 8:
            tool_names += ", ..."
        typer.secho(
            f"Beemboy — model={settings.llama_model} tools={len(mcp.tools)}. "
            "Commands: /quit /exit",
            fg=typer.colors.GREEN,
        )
        if settings.camera_enabled:
            typer.secho(
                "Camera identity enabled. Debug commands: "
                "/camera-embedding <v1,v2,...> | /camera-frame <image_path>",
                dim=True,
                err=True,
            )
        if mcp.tools:
            typer.secho(f"MCP tools loaded: {tool_names}", dim=True, err=True)
        turn_index = 0
        while True:
            try:
                line = await asyncio.to_thread(input, "> ")
            except EOFError:
                break
            if not line.strip():
                continue
            if line.strip() in {"/quit", "/exit", "exit"}:
                break
            if line.startswith("/camera-embedding "):
                if not settings.camera_enabled:
                    typer.secho("Camera identity is disabled in settings.", fg=typer.colors.YELLOW, err=True)
                    continue
                try:
                    emb = _parse_embedding_arg(line[len("/camera-embedding ") :].strip())
                    status = orch.observe_camera_embedding(emb)
                    typer.secho(f"[camera] embedding observed: {status}", fg=typer.colors.CYAN, err=True)
                except ValueError as e:
                    typer.secho(f"[camera] invalid embedding: {e}", fg=typer.colors.RED, err=True)
                continue
            if line.startswith("/camera-frame "):
                if not settings.camera_enabled:
                    typer.secho("Camera identity is disabled in settings.", fg=typer.colors.YELLOW, err=True)
                    continue
                image_path = line[len("/camera-frame ") :].strip()
                try:
                    frame_bytes = Path(image_path).expanduser().read_bytes()
                    statuses = orch.observe_camera_frame(frame_bytes)
                    typer.secho(
                        f"[camera] frame observed statuses={statuses or ['no-face']}",
                        fg=typer.colors.CYAN,
                        err=True,
                    )
                except OSError as e:
                    typer.secho(f"[camera] failed to read frame: {e}", fg=typer.colors.RED, err=True)
                continue
            try:
                turn_index += 1
                used_tool_this_turn = False
                turn_telemetry: TurnTelemetry | None = None
                telemetry_cb = None
                if telemetry_enabled:
                    def _capture_telemetry(value: TurnTelemetry) -> None:
                        nonlocal turn_telemetry
                        turn_telemetry = value

                    telemetry_cb = _capture_telemetry
                if settings.stream_responses:

                    def _on_tool_phase() -> None:
                        print(file=sys.stdout)
                        typer.secho("Using tools…", dim=True, err=True)

                    def _on_tool_call(name: str, args_json: str) -> None:
                        nonlocal used_tool_this_turn
                        used_tool_this_turn = True
                        preview = args_json if len(args_json) <= 180 else args_json[:177] + "..."
                        typer.secho(f"[MCP] {name} {preview}", fg=typer.colors.CYAN, err=True)

                    history, reply = await orch.run_turn(
                        history,
                        line,
                        on_text_delta=lambda s: print(s, end="", flush=True),
                        on_tool_round_start=_on_tool_phase,
                        on_tool_call=_on_tool_call,
                        on_telemetry=telemetry_cb,
                    )
                    print()
                else:
                    def _on_tool_call_no_stream(name: str, args_json: str) -> None:
                        nonlocal used_tool_this_turn
                        used_tool_this_turn = True
                        preview = args_json if len(args_json) <= 180 else args_json[:177] + "..."
                        typer.secho(f"[MCP] {name} {preview}", fg=typer.colors.CYAN, err=True)

                    history, reply = await orch.run_turn(
                        history,
                        line,
                        on_tool_call=_on_tool_call_no_stream,
                        on_telemetry=telemetry_cb,
                    )
                    print(reply)
                if mcp.tools and not used_tool_this_turn:
                    typer.secho("[MCP] no tool calls this turn", dim=True, err=True)
                if telemetry_enabled and turn_telemetry is not None:
                    _emit_turn_telemetry(turn_index, turn_telemetry)
            except Exception as e:
                log.exception("chat.turn_failed", error=str(e))
                typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)


async def _voice_loop(settings: Settings, *, telemetry_enabled: bool = False) -> None:
    from beemboy.voice.pipeline import VoiceAssistantLoop, VoiceLoopEvents

    try:
        servers = settings.resolved_mcp_servers()
    except (json.JSONDecodeError, ValueError) as e:
        typer.secho(f"Invalid MCP_SERVERS or MCP config: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from e
    log = structlog.get_logger("voice")
    async with MCPBundle(servers) as mcp:
        llm = LlamaServerBackend(settings)
        orch = AgentOrchestrator(settings, llm, mcp)

        def _status(msg: str) -> None:
            typer.secho(msg, fg=typer.colors.CYAN)

        def _voice_telemetry(turn_index: int, t: TurnTelemetry, v: VoiceTurnTelemetry) -> None:
            if telemetry_enabled:
                _emit_voice_telemetry(turn_index, t, v)

        loop = VoiceAssistantLoop(
            settings,
            orch,
            events=VoiceLoopEvents(on_status=_status, on_telemetry=_voice_telemetry),
        )
        typer.secho(
            f"Beemboy voice mode — model={settings.llama_model}. "
            "Wake phrase: 'Beemboy'. Ctrl+C to quit.",
            fg=typer.colors.GREEN,
        )
        try:
            await loop.run()
        except Exception as e:
            log.exception("voice.loop_failed", error=str(e))
            typer.secho(f"Voice error: {e}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from e


def _ui_loop(settings: Settings) -> None:
    """Desktop camera + identity UI."""
    from beemboy.ui.app import DesktopUIApp

    app_settings = settings
    orch = AgentOrchestrator(app_settings, _NoopLLM(), _NoopMCP())
    controller = UIController(orch)
    ui = DesktopUIApp(app_settings, controller)
    ui.run()


@app.command("chat")
def chat(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging (also enables telemetry)"),
    telemetry: bool = typer.Option(
        False,
        "--telemetry",
        help="Print per-turn latency and token estimates to stderr",
    ),
) -> None:
    """Interactive REPL against llama-server with MCP tools."""
    _configure_logging(verbose)
    settings = get_settings()
    telemetry_enabled = telemetry or verbose
    try:
        asyncio.run(_chat_loop(settings, telemetry_enabled=telemetry_enabled))
    except KeyboardInterrupt:
        typer.secho("\nBye.", dim=True)


@app.command("voice")
def voice(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging (also enables telemetry)"),
    telemetry: bool = typer.Option(
        False,
        "--telemetry",
        help="Print per-turn telemetry including STT/TTS latency to stderr",
    ),
) -> None:
    """Offline voice loop (wake word + whisper STT + piper TTS)."""
    _configure_logging(verbose)
    settings = get_settings()
    telemetry_enabled = telemetry or verbose
    try:
        asyncio.run(_voice_loop(settings, telemetry_enabled=telemetry_enabled))
    except KeyboardInterrupt:
        typer.secho("\nBye.", dim=True)


@app.command("ui")
def ui(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging"),
) -> None:
    """Desktop UI for camera preview, identities, and safe persistence controls."""
    _configure_logging(verbose)
    settings = get_settings()
    try:
        _ui_loop(settings)
    except KeyboardInterrupt:
        typer.secho("\nBye.", dim=True)


if __name__ == "__main__":
    app()
