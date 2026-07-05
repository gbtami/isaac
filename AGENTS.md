# AGENTS.md

## Project Overview
isaac ships both an ACP agent and an ACP client. The agent (`isaac.agent`) implements the Agent Client Protocol end-to-end (init/version negotiation, session lifecycle, prompt turns, tool calls, file system and terminal handling, session modes/slash commands, and MCP toolsets). The client (`isaac.client`) is an interactive REPL example that speaks ACP over stdio to any ACP-compliant agent. Core tech:
- **logic**: pydantic-ai https://github.com/pydantic/pydantic-ai
- **communication**: https://github.com/agentclientprotocol/python-sdk
- **protocol**: Agent Client Protocol https://agentclientprotocol.com/

## Protocol Compliance (must follow ACP)
- Both the agent and client must strictly follow the ACP specification so they interoperate with any other ACP-compliant client/agent. Do not introduce behavior that assumes a proprietary peer. The codebase tracks ACP Python SDK `0.10.x`, so use the official `acp.schema`, `acp.helpers`, `acp.agent` / `acp.client`, and `run_agent` / process-spawn helpers where possible. Prefer SDK contrib helpers for session/tool/permission bookkeeping when they remove local state-machine code.
- Keep initialization/version negotiation aligned with `PROTOCOL_VERSION`, honor advertised capabilities, and preserve ACP-defined session, prompt, tool call, file system, terminal, and session config option flows.
- Mode/model selection must use ACP Session Config Options (`config_options`, `config_option_update`, `session/set_config_option`) rather than custom ext methods.

## Components and Entrypoints
- Agent: `uv run isaac` or `python -m isaac` starts the ACP agent server defined in `isaac.agent`.
- Client: `uv run python -m isaac.client uv run isaac` launches the ACP client REPL against the agent; the client can also be pointed at any other ACP agent binary.

## Project Setup
This project uses `uv` for environment and project management.
- Install dependencies: `uv pip install -e .`
- Refresh offline models.dev snapshot + generated model catalog: `uv run python scripts/update_models_dev_catalog.py --top-per-provider 25`

## Cross-client Testing (install to user site)
To test isaac with other ACP clients after code changes without bumping the version install isaac to your user site and run it from there:
- Install: delete older wheels first (`rm -f dist/isaac_acp-*.whl`), then `uv build --wheel`, then `python -m pip install --user --no-deps --force-reinstall dist/isaac_acp-*.whl` to ensure the wildcard matches only the newly built wheel.
- Run with another client: `cd ~/toad && uv run toad acp "isaac" --project-dir ~/playground`

## Environment variables
- isaac loads environment variables from a shared `.env` in the platform config dir (`platformdirs`; Linux example: `~/.config/isaac/.env`). Place provider keys (e.g., `OPENROUTER_API_KEY`) there to use any cwd.
- `ISAAC_ACP_STDIO_BUFFER_LIMIT_BYTES` optionally overrides ACP stdio pipe buffer limits for both `uv run isaac` and `python -m isaac.client ...` (defaults to `52428800` / 50MB).

## Required Checks (run every change)
- Format: `uv run ruff format .`
- Lint: `uv run ruff check .`
- Types: `uv run mypy src tests`
- Tests: `uv run pytest`
- Package install check: delete older wheels first (`rm -f dist/isaac_acp-*.whl`), then `uv build --wheel`, then `python -m pip install --user --no-deps --force-reinstall dist/isaac_acp-*.whl`

## Tooling (pydantic-ai)
- Target Pydantic AI 2.x APIs. Prefer composable capabilities over ad-hoc constructor hooks or prompt-handler callbacks.
- All tool functions must take `RunContext[...]` as the first argument. Attach Isaac tools through `build_isaac_tools_capability()` at agent construction time; do not use constructor `toolsets` for normal/session tools or post-construction tool registration shims.
- `src/isaac/agent/capabilities.py` assembles Isaac-specific Pydantic AI capabilities using the public capability helpers such as `ReinjectSystemPrompt`, `ProcessHistory`, `PrepareTools`, `ProcessEventStream`, and `HandleDeferredToolCalls`. Add new cross-cutting behavior there first instead of growing `PromptHandler` or `stream_with_runner`.
- Server-side system prompts are authoritative. Keep prompt reinjection on Pydantic AI's `ReinjectSystemPrompt(replace_existing=True)` path instead of preserving stale system prompt parts from ACP/UI history.
- Provider-bound message-history cleanup should stay on the Pydantic AI `ProcessHistory` capability path, not in deprecated constructor hooks or client-specific prompt code.
- ACP-provided MCP toolsets should be wrapped via `build_toolset_capabilities()` and passed as capabilities, not through a separate `Agent(toolsets=...)` construction path.
- Transient follow-up hints such as recent files touched should be per-run Pydantic AI instructions/capabilities, not persisted chat-history mutations. ACP-facing tool/plan event projection should be attached through `ProcessEventStream` capabilities rather than `stream_with_runner` callbacks.
- `src/isaac/agent/tools/registration.py` owns the Isaac tool wrapper functions, Pydantic AI `Tool` objects, and ACP-compatible metadata. Keep tool names and argument schemas stable unless intentionally changing the ACP-visible contract.
- Required tool args are enforced through the pydantic argument models in `src/isaac/agent/tools/executor.py`; invalid direct ACP calls return an error, while invalid model tool calls raise Pydantic AI retry prompts.
- Pydantic AI Harness is available through the optional `harness` extra for experiments. Keep high-impact behavior such as CodeMode opt-in until approval, sandboxing, and ACP UX are reviewed. Experimental Harness FileSystem/Shell tools must stay behind environment flags and use prefixed `harness_*` names so they do not change the public ACP tool contract.

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
