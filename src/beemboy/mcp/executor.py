"""Tool result formatting for MCP ``call_tool`` responses."""

from __future__ import annotations

from typing import Any

from mcp.types import CallToolResult, TextContent


def format_tool_result(result: CallToolResult) -> str:
    parts: list[str] = []
    for block in result.content or []:
        if isinstance(block, TextContent):
            parts.append(block.text)
        else:
            parts.append(str(block))
    sc = result.structuredContent
    if sc is not None:
        parts.append(str(sc))
    if result.isError:
        return "Tool reported error: " + ("\n".join(parts) if parts else "unknown")
    return "\n".join(parts) if parts else "(empty tool result)"


__all__ = ["format_tool_result"]
