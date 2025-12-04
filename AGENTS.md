# AGENTS.md

## Project Overview
isaac ships both an ACP agent and an ACP client. The agent (`isaac.agent`) implements the Agent Client Protocol end-to-end (init/version negotiation, session lifecycle, prompt turns, tool calls, file system and terminal handling, session modes/slash commands, and MCP toolsets). The client (`isaac.client`) is an interactive REPL example that speaks ACP over stdio to any ACP-compliant agent. Core tech:
- **logic**: pydantic-ai https://github.com/pydantic/pydantic-ai
- **communication**: Agent Client Protocol https://agentclientprotocol.com/overview/introduction via https://github.com/agentclientprotocol/python-sdk

## Protocol Compliance (must follow ACP)
- Both the agent and client must strictly follow the ACP specification so they interoperate with any other ACP-compliant client/agent. Do not introduce behavior that assumes a proprietary peer. The codebase tracks the ACP Python SDK `0.7.x`, so use the snake_case schema fields and the `run_agent` / `connect_to_agent` helpers.
- Keep initialization/version negotiation aligned with `PROTOCOL_VERSION`, honor advertised capabilities, and preserve ACP-defined session, prompt, tool call, file system, terminal, and session mode flows.
- When adding features, favor ACP extension points (e.g., extMethods such as `model/list` and `model/set`) instead of one-off custom channels.

## Components and Entrypoints
- Agent: `uv run isaac` or `python -m isaac` starts the ACP agent server defined in `isaac.agent`.
- Client: `uv run python -m isaac.client uv run isaac` launches the ACP client REPL against the agent; the client can also be pointed at any other ACP agent binary.

## Project Setup
This project uses `uv` for environment and project management.
- Install dependencies: `uv pip install -e .`

## Required Checks (run every change)
- Format: `uv run ruff format .`
- Lint: `uv run ruff check .`
- Tests: `uv run pytest`
