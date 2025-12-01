"""MCP server helper functions for ACP session setup.

Converts ACP `mcpServers` entries into pydantic-ai MCP toolsets so the agent can
attach to MCP servers over stdio/HTTP/SSE as described in the ACP Session Setup
and Transports sections.
"""

from __future__ import annotations

from typing import Any, List

from pydantic_ai import mcp as mcp_client  # type: ignore


def build_mcp_toolsets(servers: list[Any] | None) -> List[Any]:
    """Construct pydantic-ai MCP toolsets from ACP session mcpServers config."""
    toolsets: list[Any] = []
    for server in servers or []:
        if isinstance(server, dict):
            stype = server.get("type")
            name = server.get("name") or None
            headers = server.get("headers")
            env_list = server.get("env")
            args = server.get("args")
            command = server.get("command")
            url = server.get("url")
        else:
            stype = getattr(server, "type", None)
            name = getattr(server, "name", "") or None
            headers = getattr(server, "headers", None)
            env_list = getattr(server, "env", None)
            args = getattr(server, "args", None)
            command = getattr(server, "command", None)
            url = getattr(server, "url", None)
        prefix = name.replace(" ", "_") if name else None
        try:
            if stype == "stdio":
                if not command:
                    continue
                env = (
                    {ev["name"]: ev["value"] for ev in env_list}
                    if isinstance(env_list, list) and env_list and isinstance(env_list[0], dict)
                    else {ev.name: ev.value for ev in env_list or []}
                )
                arg_list = list(args or [])
                toolsets.append(
                    mcp_client.MCPServerStdio(
                        command,
                        arg_list,
                        env=env or None,
                        tool_prefix=prefix,
                        id=name,
                    )
                )
            elif stype == "http":
                if not url:
                    continue
                header_map = (
                    {h["name"]: h["value"] for h in headers}
                    if isinstance(headers, list) and headers and isinstance(headers[0], dict)
                    else {h.name: h.value for h in headers or []}
                )
                toolsets.append(
                    mcp_client.MCPServerStreamableHTTP(
                        url,
                        headers=header_map or None,
                        tool_prefix=prefix,
                        id=name,
                    )
                )
            elif stype == "sse":
                if not url:
                    continue
                header_map = (
                    {h["name"]: h["value"] for h in headers}
                    if isinstance(headers, list) and headers and isinstance(headers[0], dict)
                    else {h.name: h.value for h in headers or []}
                )
                toolsets.append(
                    mcp_client.MCPServerSSE(
                        url,
                        headers=header_map or None,
                        tool_prefix=prefix,
                        id=name,
                    )
                )
        except Exception:
            continue
    return toolsets
