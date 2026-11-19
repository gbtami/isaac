# Isaac

Isaac is an ACP coding server built on [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/).
Use it from [Toad](https://github.com/batrachianai/toad), [Isaac Terminal Client](client/), or another ACP client.

The agent is Harness's `Coder`, assembled with Harness planning, skills, permissions, and compaction.
Harness's [ACP adapter](https://pydantic.dev/docs/ai/harness/acp/) handles the protocol.
Isaac supplies process configuration and connects the client's workspace and MCP servers.
The server and terminal client are independently installable. They communicate over ACP stdio,
so each can be used with a compatible implementation of the other side.

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
Isaac uses Pydantic AI's providers directly, without a separate model catalog.

You can override the model with `--model`. Repeat it to populate the ACP `model` session config option;
the first is the default for each new session:

```bash
isaac --model anthropic:claude-sonnet-4-6 --model openai:gpt-5.4
```

Credentials are needed when prompting, not for initialization or creating a session.
In clients such as Toad, `/models` lists the configured choices and `/model <id>` selects one for
the session. Model selection is shared with the ACP config option and saved for resume.
The adapter initializes the current model before processing commands. If its provider setup fails,
start Isaac with a working model first.

### ChatGPT/Codex browser login

Use Pydantic AI's Codex provider with your ChatGPT/Codex subscription:

```bash
isaac login
```

This opens a browser and waits up to five minutes for Pydantic AI's localhost callback at
`http://localhost:1455/auth/callback`. The browser must be able to reach the machine running
Isaac on that port. Use `isaac login --no-browser` to open the printed URL yourself.
Login does not require `ISAAC_MODEL` and runs separately from the ACP server.

Then set a model in the shared environment, or pass it to the server:

```dotenv
ISAAC_MODEL=openai-codex:gpt-5.6-luna
```

Start Isaac from Toad or Isaac Terminal Client with the same saved login.
Choose a model available to your ChatGPT/Codex account. If a prompt returns `model_not_found`,
check the model picker in Codex and use one of the listed model IDs with the `openai-codex:` prefix.
Run `isaac login` again to replace the saved login, then restart existing Isaac processes.
Codex authentication does not use `OPENAI_API_KEY`.

Isaac stores credentials in the platform config directory, such as
`~/.config/isaac/auth/openai-codex.json` on Linux. This is an unencrypted file with private
`0600` permissions in a `0700` directory. Pydantic AI handles token refresh and writes rotated
credentials through Isaac's atomic file store. Tokens are separate from saved conversations.

Without an Isaac login, Pydantic AI can reuse an existing Codex CLI login from
`~/.codex/auth.json` (or `$CODEX_HOME/auth.json`). This fallback never writes that file;
refreshed tokens last only for the server process. Use `isaac login` for Isaac-owned persistence.

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

## Use with Isaac Terminal Client

Install the client separately from this checkout and pass the agent command to launch.

```bash
uv tool install ./client
isaac-client --project-dir ~/playground isaac
```

Use `/client:model` to select from the models Isaac advertises. Run the same command with
`--resume` before the agent command to choose a named session. See the [client guide](client/README.md)
for the client's supported ACP features and an example with another agent.

## Coding capabilities

- `Coder`: `read_file`, `write_file`, `edit_file`, `list_files`, `grep`, persistent `shell`, and self-delegation.
- Delegation: Harness lets the model run delegates in the background, list them, send instructions,
  answer their questions, and cancel them. Repeat `--agent-dir` to load named delegates from folders
  in each session's workspace.
- Repository instructions: Harness loads `AGENTS.md` / `CLAUDE.md` through Coder's `RepoContext`.
- Planning: Harness's `write_plan` tool, with a separate persisted plan per ACP session and native plan sidebar updates.
- Skills: `.agents/skills` in the session workspace. Repeat `--skill-dir` to choose other libraries.
- Context management: Coder clears older tool results. Automatic compaction uses the provider's
  inline support on Codex, OpenAI Responses, and supported Claude models on the Anthropic API.
  It targets 85% of the context window. Harness summarizes and trims on other endpoints or when
  the conversation is already near the limit. `--context-window TOKENS` overrides window resolution.
  Toad's status line shows estimated context usage before model requests when the window size is known.
- Permissions: Harness `Permissions`, with approval decisions handled by the ACP client.
- Structured questions: Harness `AskUser` sends ACP forms when the client advertises form elicitation.
- MCP: ACP-provided stdio, HTTP, and SSE servers, with server-prefixed tool names.
- Code Mode: Harness exposes eligible tools through Monty's `run_code` by default. Pass
  `--no-code-mode` to use direct tool calls.
  When a nested operation needs Harness approval, the ACP client offers **Allow once** or **Reject**
  before the operation runs.

The default `--permissions auto` permits file changes and ordinary shell commands; Harness asks about
commands matching its built-in sensitive-command patterns. `--permissions read-only` asks before every
file change and shell command. `--permissions full` allows those operations without asking.
These are Harness policies, not an OS sandbox. File tools are rooted at the workspace; shell commands
run locally with the server process's privileges. MCP tools follow their server's own behavior.
Self-delegates inherit the coding agent's capabilities. The main session owns its plan and command UI.
A delegate that needs approval stops and reports back to
the parent; the parent can perform that operation through ACP approval.

Background tasks belong to the current agent run. The parent can work while they run and can act
on their reports before finishing. Cancelling the run stops its delegates. A direct ACP approval
can end that run and start another, so tasks do not carry across that pause or a resumed session.

For example, `isaac --agent-dir .agents/agents` loads Harness's [Markdown and TOML delegate definitions](https://pydantic.dev/docs/ai/harness/subagents/#loading-sub-agents-from-disk).
Named delegates use the current model and the instructions in their definition. They have no tools of
their own and can ask the parent questions through `ask_parent`. The model can delegate to `self` for
work that uses Isaac's coding tools.

ACP supports streaming, cancellation, concurrent sessions, tool presentation, and model config options
through the upstream adapter. A small session capability adds standard ACP plan updates and these commands:

- `/help`: list the agent commands.
- `/plan`: show the current session's plan and refresh the client's plan sidebar.
- `/title`: show the saved session title.
- `/models`: list the configured model IDs and mark the current model.
- `/model [id]`: show the current model or switch to one of the listed IDs for subsequent turns.
- `/permissions [auto|read-only|full]`: show or change Harness permissions for the live session.
  On resume the permission mode starts from `--permissions` again.
- `/compact`: use native compaction on supported OpenAI Responses and Anthropic endpoints. If native
  compaction is unavailable or fails, use Harness summarization and fall back to trimming.
- `/compact <focus>`: use Harness summarization and tell it what to preserve.

Codex subscription and Gemini models use Harness for manual `/compact`. Anthropic conversations
with a signed native `/compact` block also use Harness for automatic compaction while the block is
in history, because the API rejects the inline compaction edit alongside it.

The commands are advertised to ACP clients and handled through public capability hooks. Ordinary prompts,
including unknown slash commands and messages with attachments, continue through the normal agent path.
Cancelled plan steps are omitted from the native sidebar because ACP has no cancelled entry status.
The upstream ACP adapter is experimental and is pinned to Harness's 0.69 release series.

## Saved sessions and resume

Sessions save automatically when created, after each completed turn, and when the selected model changes.
The stored snapshot contains model history, the client-visible transcript, and the selected model.
Plans use Harness's SQLite plan store and are restored into the native sidebar on load.

On Linux the database is `~/.local/state/isaac/sessions.sqlite3`; other platforms use their standard
user state directory. Set `ISAAC_STATE_DIR` in the environment or shared `.env` to choose another directory.
The database is private to your user and contains conversation and tool content.

In **Toad**, press **Ctrl+R** to select a previous session. Isaac advertises `loadSession`, so Toad
records new Isaac sessions in its own history and reloads them through standard ACP `session/load`.
Resume restores the conversation without rerunning past tools. Provider credentials must still be
available, and the selected model should remain in the server's configured model list.

Isaac generates a short title from the first coding prompt using one separate, tool-free request to
the selected model. This request is limited to eight seconds; if it fails, a shortened first prompt
becomes the title. Local slash commands do not name a session. Titles remain stable across later
turns and restarts, and older unnamed sessions are named when next used.

Titles are published through standard ACP `session_info_update` notifications. Clients that support
these notifications can use them in their UI. The current local Toad clone ignores them, so its resume
picker will still show its own names. Use `/title` to see Isaac's generated name and, if desired,
Toad's existing `/toad:rename <name>` command to set that name locally. Isaac never modifies Toad's
source or writes its database.

Harness also saves checkpoints during a turn. If Isaac exits before the turn is saved, reopening
the session restores the latest available checkpoint into the model's context and shows a recovery notice.
Ask Isaac to continue when ready. Check the workspace for partial changes before retrying writes
or commands. Reopening a session does not rerun tools or restart shell jobs.
The upstream adapter does not implement `session/list` or the separate `session/resume` method;
Toad's own history picker supplies discovery and uses `session/load` for continuation.

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
