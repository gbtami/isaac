# AGENTS.md

## Project Overview
isaac ships both an ACP agent and an ACP client. The agent (`isaac.agent`) implements the Agent Client Protocol end-to-end (init/version negotiation, session lifecycle, prompt turns, tool calls, file system and terminal handling, session modes/slash commands, and MCP toolsets). The client (`isaac.client`) is an interactive REPL example that speaks ACP over stdio to any ACP-compliant agent. Core tech:
- **logic**: pydantic-ai https://github.com/pydantic/pydantic-ai
- **communication**: https://github.com/agentclientprotocol/python-sdk
- **protocol**: Agent Client Protocol https://agentclientprotocol.com/

## Protocol Compliance (must follow ACP)
- Both the agent and client must strictly follow the ACP specification so they interoperate with any other ACP-compliant client/agent. Do not introduce behavior that assumes a proprietary peer. The codebase tracks ACP Python SDK `0.8.0`, so use snake_case schema fields and the `run_agent` / `connect_to_agent` helpers.
- Keep initialization/version negotiation aligned with `PROTOCOL_VERSION`, honor advertised capabilities, and preserve ACP-defined session, prompt, tool call, file system, terminal, and session config option flows.
- Mode/model selection must use ACP Session Config Options (`config_options`, `config_option_update`, `session/set_config_option`) rather than custom ext methods.

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
- Package install check: `uv build --wheel` then `python -m pip install --user --no-deps --force-reinstall dist/isaac-*.whl`

## Tooling (pydantic-ai)
- All tool functions must take `RunContext[...]` as the first argument; registration uses the public `Agent.tool` decorator (no private attributes).
- `register_tools` in `src/isaac/agent/tools/registration.py` binds `ctx` automatically and centralizes tool registration.
- Required tool args are enforced in `run_tool` in `src/isaac/agent/tools/executor.py`; missing args return an error instead of calling the handler.

## Code Structure (responsibilities)
- `src/isaac/agent/` — ACP agent implementation (session lifecycle, prompt handling, tool calls, filesystem/terminal endpoints, slash commands, model registry). Key files:
  - `acp/`: ACP endpoint handlers split by domain (init, permissions, sessions, prompts, tools, filesystem, terminal, updates, extensions).
  - `acp_agent.py`: ACP-facing agent composed from ACP handler mixins.
  - `brain/prompt_handler.py`: Prompt handling with plan/tool integration; assistant text is emitted once at end-of-turn (tool/plan/thought updates may still stream).
  - `brain/prompt_runner.py`: Stream handling and tool call updates.
  - `brain/plan_parser.py`: Plan parsing utilities for converting model text to ACP plan updates.
  - `brain/plan_updates.py`: Plan update helpers (status updates, stable IDs).
  - `brain/agent_factory.py`: Model runner creation for prompt handling.
  - `brain/tool_args.py`: Tool argument coercion for model calls.
  - `plan_shortcuts.py`: Parse `plan:` prompt shortcuts into ACP plan updates.
  - `models.py`: Model registry/config loader; builds pydantic-ai runners.
  - `subagents/`: Delegate agents exposed as tools with isolated context; contributor guide in `src/isaac/agent/subagents/CONTRIBUTOR.md`.
  - `tools/`: Tool implementations plus registry/schema/executor/registration modules.
  - `tool_execution.py`: ACP tool execution helpers (including terminal-backed run_command).
  - `runner.py`: Streaming utilities for pydantic-ai runs.
  - `prompt_utils.py`: Helpers for extracting user text blocks.
  - `slash.py`: Server-side slash command registry/handlers (`/log`, `/model`, `/models`, `/usage`).
- `src/isaac/client/` — ACP client REPL example. Key files:
  - `repl.py`: Interactive loop and prompt submission; delegates slash commands to `client/slash.py`.
  - `slash.py`: Client-side slash command registry/handlers (local-only commands) and help rendering; forwards unknown/agent commands.
  - `acp_client.py`: ACP client implementation; handles session updates (config option updates, available commands, tool updates, agent messages).
  - `session_state.py`: Shared REPL/UI state.
  - `status_box.py`, `display.py`, `thinking.py`: Rendering helpers for status, text, and thinking traces.
