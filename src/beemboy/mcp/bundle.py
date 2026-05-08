from __future__ import annotations

import json
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

import structlog
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client

from beemboy.config.settings import HttpMCPServer, StdioMCPServer
from beemboy.mcp.executor import format_tool_result

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class RegisteredTool:
    server_id: str
    mcp_name: str
    openai_name: str
    session: ClientSession
    description: str
    parameters_schema: dict[str, Any]

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.openai_name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }


class MCPBundle:
    """Connects MCP servers, registers namespaced tools, executes tool calls."""

    def __init__(self, servers: list[StdioMCPServer | HttpMCPServer]) -> None:
        self._servers = servers
        self.tools: list[RegisteredTool] = []
        self._by_openai: dict[str, RegisteredTool] = {}
        self._stack: AsyncExitStack | None = None

    async def __aenter__(self) -> MCPBundle:
        stack = AsyncExitStack()
        await stack.__aenter__()
        self._stack = stack
        try:
            sessions: dict[str, ClientSession] = {}
            for cfg in self._servers:
                if isinstance(cfg, StdioMCPServer):
                    import os

                    env = os.environ.copy()
                    env.update(cfg.env)
                    params = StdioServerParameters(
                        command=cfg.command,
                        args=list(cfg.args),
                        env=env,
                    )
                    read_write = await stack.enter_async_context(stdio_client(params))
                    read, write = read_write
                else:
                    if cfg.http_mode == "sse":
                        read_write = await stack.enter_async_context(sse_client(cfg.url))
                        read, write = read_write
                    else:
                        conn = await stack.enter_async_context(streamable_http_client(cfg.url))
                        read, write, _session_cb = conn

                sess_cm = ClientSession(read, write)
                session = await stack.enter_async_context(sess_cm)
                await session.initialize()
                sessions[cfg.id] = session
                log.info("mcp.connected", server_id=cfg.id)

            await self._register_tools(sessions)
            return self
        except BaseException:
            await stack.__aexit__(*sys.exc_info())
            self._stack = None
            raise

    async def __aexit__(self, *args: Any) -> bool | None:
        if self._stack:
            await self._stack.__aexit__(*args)
            self._stack = None
        self.tools.clear()
        self._by_openai.clear()
        return None

    async def _register_tools(self, sessions: dict[str, ClientSession]) -> None:
        for server_id, session in sessions.items():
            listed = await session.list_tools()
            for tool in listed.tools:
                openai_name = f"{server_id}__{tool.name}"
                desc = (tool.description or "").strip()
                schema = tool.inputSchema
                if not isinstance(schema, dict):
                    schema = {"type": "object", "properties": {}}
                rt = RegisteredTool(
                    server_id=server_id,
                    mcp_name=tool.name,
                    openai_name=openai_name,
                    session=session,
                    description=desc or f"MCP tool {tool.name} from {server_id}",
                    parameters_schema=schema,
                )
                self.tools.append(rt)
                self._by_openai[openai_name] = rt
                log.debug("mcp.tool_registered", openai_name=openai_name)

    async def invoke(self, openai_name: str, arguments_json: str) -> str:
        rt = self._by_openai.get(openai_name)
        if not rt:
            return f"Unknown tool: {openai_name}"
        try:
            args = json.loads(arguments_json) if arguments_json.strip() else {}
        except json.JSONDecodeError as e:
            return f"Invalid tool arguments JSON: {e}"
        if not isinstance(args, dict):
            return "Tool arguments must be a JSON object."
        try:
            result = await rt.session.call_tool(rt.mcp_name, arguments=args)
        except Exception as e:
            log.exception("mcp.call_tool_failed", tool=openai_name)
            return f"Tool error ({openai_name}): {e}"
        return format_tool_result(result)
