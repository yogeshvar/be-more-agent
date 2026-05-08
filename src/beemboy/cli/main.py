from __future__ import annotations

import asyncio
import json
import sys

import structlog
import typer

from beemboy.agent.orchestrator import AgentOrchestrator
from beemboy.config.settings import Settings, get_settings
from beemboy.llm.llama_server import LlamaServerBackend
from beemboy.mcp.bundle import MCPBundle

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


async def _chat_loop(settings: Settings) -> None:
    try:
        servers = settings.resolved_mcp_servers()
    except (json.JSONDecodeError, ValueError) as e:
        typer.secho(f"Invalid MCP_SERVERS or MCP config: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from e
    log = structlog.get_logger("cli")
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
        if mcp.tools:
            typer.secho(f"MCP tools loaded: {tool_names}", dim=True, err=True)
        while True:
            try:
                line = await asyncio.to_thread(input, "> ")
            except EOFError:
                break
            if not line.strip():
                continue
            if line.strip() in {"/quit", "/exit", "exit"}:
                break
            try:
                if settings.stream_responses:

                    def _on_tool_phase() -> None:
                        print(file=sys.stdout)
                        typer.secho("Using tools…", dim=True, err=True)

                    def _on_tool_call(name: str, args_json: str) -> None:
                        preview = args_json if len(args_json) <= 180 else args_json[:177] + "..."
                        typer.secho(f"[MCP] {name} {preview}", fg=typer.colors.CYAN, err=True)

                    history, reply = await orch.run_turn(
                        history,
                        line,
                        on_text_delta=lambda s: print(s, end="", flush=True),
                        on_tool_round_start=_on_tool_phase,
                        on_tool_call=_on_tool_call,
                    )
                    print()
                else:
                    def _on_tool_call_no_stream(name: str, args_json: str) -> None:
                        preview = args_json if len(args_json) <= 180 else args_json[:177] + "..."
                        typer.secho(f"[MCP] {name} {preview}", fg=typer.colors.CYAN, err=True)

                    history, reply = await orch.run_turn(history, line, on_tool_call=_on_tool_call_no_stream)
                    print(reply)
            except Exception as e:
                log.exception("chat.turn_failed", error=str(e))
                typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)


@app.command("chat")
def chat(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging"),
) -> None:
    """Interactive REPL against llama-server with MCP tools."""
    _configure_logging(verbose)
    settings = get_settings()
    try:
        asyncio.run(_chat_loop(settings))
    except KeyboardInterrupt:
        typer.secho("\nBye.", dim=True)


if __name__ == "__main__":
    app()
