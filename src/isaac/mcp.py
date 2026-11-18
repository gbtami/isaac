"""Connect ACP-provided MCP servers using Pydantic AI's public MCP toolset."""

import re

from acp import schema
from fastmcp.client.transports import ClientTransport, SSETransport, StdioTransport, StreamableHttpTransport
from pydantic_ai.capabilities import Toolset
from pydantic_ai.mcp import MCPToolset
from pydantic_ai_harness.experimental.acp import ACPSession


def mcp_capabilities(session: ACPSession) -> list[Toolset[None]]:
    """Keep MCP connections and tool names separate for each session and server."""
    capabilities: list[Toolset[None]] = []
    prefixes: set[str] = set()
    for server in session.mcp_servers:
        prefix = re.sub(r"[^a-zA-Z0-9_]", "_", server.name)
        if not prefix or prefix in prefixes:
            raise ValueError("MCP server names must produce distinct, nonempty tool prefixes.")
        prefixes.add(prefix)
        transport: ClientTransport
        if isinstance(server, schema.McpServerStdio):
            transport = StdioTransport(
                command=server.command,
                args=server.args,
                env={entry.name: entry.value for entry in server.env} or None,
                cwd=session.cwd,
            )
        elif isinstance(server, schema.HttpMcpServer):
            transport = StreamableHttpTransport(
                server.url, headers={entry.name: entry.value for entry in server.headers}
            )
        elif isinstance(server, schema.SseMcpServer):
            transport = SSETransport(server.url, headers={entry.name: entry.value for entry in server.headers})
        else:
            raise ValueError(f"Unsupported MCP transport for {server.name!r}.")
        toolset = MCPToolset[None](transport, id=server.name).prefixed(prefix)
        capabilities.append(Toolset(toolset))
    return capabilities
