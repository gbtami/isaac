# AGENTS.md

## Project Overview
isaac ships both an ACP agent and an ACP client. The agent (`isaac.agent`) implements the Agent Client Protocol end-to-end (init/version negotiation, session lifecycle, prompt turns, tool calls, file system and terminal handling, session modes/slash commands, and MCP toolsets). The client (`isaac.client`) is an interactive REPL example that speaks ACP over stdio to any ACP-compliant agent. Core tech:
- **logic**: pydantic-ai https://github.com/pydantic/pydantic-ai
- **communication**: https://github.com/agentclientprotocol/python-sdk
- **protocol**: Agent Client Protocol https://agentclientprotocol.com/

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

## Cross-client Testing (install to user site)
To test isaac with other ACP clients after code changes without bumping the version install isaac to your user site and run it from there:
- Install: `uv build --wheel` then `python -m pip install --user --no-deps --force-reinstall dist/isaac-*.whl` to ensure the new code is picked up.
- Run with another client: `cd ~/toad && uv run toad acp "isaac" --project-dir ~/playground`

## Environment variables
- isaac loads a shared `.env` from `~/.config/isaac/.env`, then a `.env` in the current working directory (latter can override). Place provider keys (e.g., `OPENROUTER_API_KEY`) in the config path to use any cwd.

## Required Checks (run every change)
- Format: `uv run ruff format .`
- Lint: `uv run ruff check .`
- Types: `uv run mypy src tests`
- Tests: `uv run pytest`

## Tooling (pydantic-ai)
- All tool functions must take `RunContext[...]` as the first argument; registration uses the public `Agent.tool` decorator (no private attributes).
- `register_tools` in `src/isaac/agent/runner.py` wraps handlers so they bind `ctx` automatically and filter unexpected args.
- Required tool args are enforced in `run_tool`; missing args return an error instead of calling the handler.

## Code Structure (responsibilities)
- `src/isaac/agent/` — ACP agent implementation (session lifecycle, prompt handling, tool calls, filesystem/terminal endpoints, slash commands, model registry). Key files:
  - `acp_agent.py`: ACP-facing agent; wiring for sessions, prompts, tools, slash commands, notifications, and the handoff planning/execution flow.
  - `brain/handoff.py`: Handoff planning/execution helpers.
  - `brain/planner.py`: Plan parsing utilities for converting model text to ACP plan updates.
  - `models.py`: Model registry/config loader; builds executor/planner agents.
  - `tools/`: Local tool implementations plus registry (`TOOL_HANDLERS`, `TOOL_REQUIRED_ARGS`).
  - `runner.py`: Registers tools with pydantic-ai models and streams prompts/events to ACP updates.
  - `prompt_utils.py`: Helpers for extracting user text blocks.
  - `slash.py`: Server-side slash command registry/handlers (`/log`, `/model`, `/models`, `/usage`).
- `src/isaac/client/` — ACP client REPL example. Key files:
  - `repl.py`: Interactive loop and prompt submission; delegates slash commands to `client/slash.py`.
  - `slash.py`: Client-side slash command registry/handlers (local-only commands) and help rendering; forwards unknown/agent commands.
  - `acp_client.py`: ACP client implementation; handles session updates (mode changes, available commands, tool updates, agent messages).
  - `session_state.py`: Shared REPL/UI state.
  - `status_box.py`, `display.py`, `thinking.py`: Rendering helpers for status, text, and thinking traces.
