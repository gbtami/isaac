# Isaac

Isaac is an ACP coding server built on [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/).
Use it from [Toad](https://github.com/batrachianai/toad) or another ACP client.

The agent is Harness's `Coder`, assembled with Harness planning, skills, permissions, and compaction.
Harness's [ACP adapter](https://pydantic.dev/docs/ai/harness/acp/) handles the protocol.
Isaac supplies process configuration and connects the client's workspace and MCP servers.
There is no bundled chat client, CLAI2 dependency, plugin framework, or custom ACP implementation.

## Install

```bash
uv tool install isaac-acp
```

For this checkout:

```bash
uv sync
uv run isaac --help
```

Python 3.11 or newer is required. Harness's `coder` extra installs the file-search dependency.

## Configure a model

Set `ISAAC_MODEL` to a Pydantic AI `provider:model` identifier and supply that provider's credentials.
Isaac loads the shared `.env` at the platform config directory, such as `~/.config/isaac/.env` on Linux.
Variables already present in the environment take precedence. Workspace `.env` files are not loaded.

For example:

```dotenv
ISAAC_MODEL=anthropic:claude-sonnet-4-6
ANTHROPIC_API_KEY=...
```

OpenAI, Anthropic, Google, and OpenAI-compatible providers such as OpenRouter are included.
Use the provider environment variables and model identifiers documented by Pydantic AI.
Other providers may require installing their `pydantic-ai-slim` extras into the same environment.
Isaac has no separate model catalog or subscription/OAuth implementation.

You can override the model with `--model`. Repeat it to populate the ACP `model` session config option;
the first is the default for each new session:

```bash
isaac --model anthropic:claude-sonnet-4-6 --model openai:gpt-5.4
```

Credentials are needed when prompting, not for initialization or creating a session.

## Use with Toad

With the installed `isaac` command and model configured in the shared environment:

```bash
cd ~/toad
uv run toad acp "isaac" --project-dir ~/playground
```

To run directly from this checkout without installing it as a tool:

```bash
cd ~/toad
uv run toad acp "uv run --directory /home/tami/isaac isaac" --project-dir ~/playground
```

Toad supplies each session's working directory. All file, shell, repository-instruction, and skill
operations use that session's workspace, even when Isaac starts in another directory.
Stdout carries ACP JSON-RPC only; diagnostics go to stderr.
`python -m isaac` starts the same server.

## Coding capabilities

- `Coder`: `read_file`, `write_file`, `edit_file`, `list_files`, `grep`, persistent `shell`, and self-delegation.
- Repository instructions: Harness loads `AGENTS.md` / `CLAUDE.md` through Coder's `RepoContext`.
- Planning: Harness's `write_plan` tool, with a separate in-memory plan per ACP session and native plan sidebar updates.
- Skills: `.agents/skills` in the session workspace. Repeat `--skill-dir` to choose other libraries.
- Context management: Coder's tool-result clearing, followed by tiered summarization
  and truncation at 85% of the model's context window. `--context-window TOKENS` overrides window resolution.
- Permissions: Harness `Permissions`, with approval decisions handled by the ACP client.
- MCP: ACP-provided stdio, HTTP, and SSE servers, with server-prefixed tool names.
- Optional `--code-mode`: Harness's Monty-backed `CodeMode` exposes eligible tools through `run_code`.

The default `--permissions auto` permits file changes and ordinary shell commands; Harness asks about
commands matching its built-in sensitive-command patterns. `--permissions read-only` asks before every
file change and shell command. `--permissions full` allows those operations without asking.
These are Harness policies, not an OS sandbox. File tools are rooted at the workspace; shell commands
run locally with the server process's privileges. MCP tools follow their server's own behavior.
Delegates inherit the coding agent's capabilities; the main session owns its plan and command UI.
A delegate that needs approval stops and reports back to
the parent; the parent can perform that operation through ACP approval.

ACP supports streaming, cancellation, concurrent sessions, tool presentation, and model config options
through the upstream adapter. A small session capability adds standard ACP plan updates and these commands:

- `/help`: list the agent commands.
- `/plan`: show the current session's plan and refresh the client's plan sidebar.
- `/compact [focus]`: summarize history with Harness compaction, falling back to truncation if summarization fails.
  An optional focus says what the summary should preserve. Summarization may use the selected model.

The commands are advertised to ACP clients and handled through public capability hooks. Ordinary prompts,
including unknown slash commands and messages with attachments, continue through the normal agent path.
Cancelled plan steps are omitted from the native sidebar because ACP has no cancelled entry status.
Sessions and plans live in memory for the connection;
Isaac does not add persistence or advertise `session/load` support.
The upstream ACP adapter is experimental and is pinned to Harness's 0.69 release series.

## Development

```bash
uv sync
uv run ruff format .
uv run ruff check .
uv run mypy src tests
uv run pytest
uv build --wheel
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the small server layout.
