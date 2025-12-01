"""Helpers for loading MCP server definitions for the client CLI."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from acp.schema import EnvVariable, HttpHeader, HttpMcpServer, SseMcpServer, StdioMcpServer


def load_mcp_config(path: str) -> list[Any]:
    """Load MCP server definitions from a JSON file into ACP schema objects."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[failed to read mcp-config: {exc}]", file=sys.stderr)
        return []

    if not isinstance(data, list):
        print("[mcp-config must be a JSON array]", file=sys.stderr)
        return []

    servers: list[Any] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        stype = entry.get("type")
        name = entry.get("name") or ""
        if stype == "stdio":
            command = entry.get("command")
            if not command:
                continue
            servers.append(
                StdioMcpServer(
                    name=name,
                    command=command,
                    args=entry.get("args", []),
                    env=[
                        EnvVariable(name=ev["name"], value=ev["value"])
                        for ev in entry.get("env", [])
                        if isinstance(ev, dict) and "name" in ev and "value" in ev
                    ],
                )
            )
        elif stype == "http":
            url = entry.get("url")
            if not url:
                continue
            servers.append(
                HttpMcpServer(
                    name=name,
                    url=url,
                    headers=[
                        HttpHeader(name=h["name"], value=h["value"])
                        for h in entry.get("headers", [])
                        if isinstance(h, dict) and "name" in h and "value" in h
                    ],
                )
            )
        elif stype == "sse":
            url = entry.get("url")
            if not url:
                continue
            servers.append(
                SseMcpServer(
                    name=name,
                    url=url,
                    headers=[
                        HttpHeader(name=h["name"], value=h["value"])
                        for h in entry.get("headers", [])
                        if isinstance(h, dict) and "name" in h and "value" in h
                    ],
                )
            )
    return servers
