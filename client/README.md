# Isaac Terminal Client

Isaac Terminal Client is a standalone terminal client for ACP agents. It launches an agent command over
stdio and uses the official ACP Python SDK. Install and configure the agent you want to use separately.

```bash
uv tool install ./client
isaac-client --project-dir ~/playground isaac
```

Use the same client with another agent, such as an installed OpenCode.

```bash
isaac-client --project-dir ~/playground opencode acp --cwd ~/playground
```

Run both components from the checkout with their separate dependencies.

```bash
uv run --project client isaac-client --project-dir ~/playground \
  uv run --project . --no-active isaac
```

Client options go before the agent command. The remaining arguments go to the agent.
The agent process starts in the caller's directory, and ACP supplies the project directory for the session.
Configure model credentials through the chosen agent.

### Use a ChatGPT/Codex subscription with Isaac

Isaac Terminal Client and Toad can use the same browser login saved by Isaac. If you need to sign in
from this checkout, run `uv run isaac login` before starting the client. Select the `openai-codex:`
provider for the subscription. The `openai:` provider uses an API key.

```bash
uv run --project client isaac-client --project-dir ~/playground \
  uv run --project . --no-active isaac --model openai-codex:gpt-5.6-luna
```

If your shared Isaac configuration already sets `ISAAC_MODEL` to a Codex model, use the command
above without `--model`.

## Terminal controls

- Type `/` to see client and agent commands. Use ↑/↓ to select and Enter to confirm. Selecting a
  command with an argument hint fills in its name so you can enter the arguments.
- Enter sends the draft, Alt+Enter inserts a newline, and multiline paste stays in the draft until
  you send it.
- `/model` opens the client model picker when the agent leaves that name available. With Isaac,
  `/model` is the agent command and `/pick-model` opens the picker.
- `/export PATH` saves a UTF-8 transcript with prompts, replies, thinking, and the tool details
  sent over ACP, including diffs. Relative paths resolve in the workspace. Choose a new file.
  For resumed sessions, the transcript includes the history supplied by the agent on load.
- `/config` opens all advertised session config options, including grouped choices and toggles.
- With Isaac, `/plan [goal]` enters read-only Plan mode. When a proposal is ready, choose
  **Implement this plan** or **Keep planning**. You can also run `/implement` to start coding,
  `/cancel-plan` to leave Plan mode, or `/checklist` to see implementation progress.
- `/help` shows available commands. When an agent has no `/plan` command, the client provides one
  to show the latest plan sent over ACP.
- `/quit` exits. Ctrl+D also exits from an empty idle prompt. While the agent is working, Enter
  queues a follow-up and Ctrl+C requests cancellation and clears the queue.

Use ↑/↓ and Enter in pickers, or type a choice number. Press Esc to close a picker. Model, setting,
and session pickers filter as you type.

The status line shows the session mode, plan progress, context usage, and the active tool. Reads and searches appear
in grouped summaries. Failures and edit diffs appear inline.

Permission requests show the supplied options and edit diffs when available. Select an option explicitly
in the terminal. Enter with no option selected, EOF and scripted input cancel the request.

When an agent asks a structured question, enter the requested answers and review them before sending.
You can edit the answers or decline the form. Isaac offers these questions to clients that advertise
ACP form elicitation.

## Start and resume sessions

Use `/new` from the idle prompt to start a fresh session in the same workspace, or `/resume` to
switch to a saved session. When the agent supports `session/close`, the client closes the previous
live session after a successful switch and closes the active session on exit.
To open the picker at startup, run the same agent command from the same project with `--resume`.

```bash
isaac-client --resume --project-dir ~/playground isaac
```

Type part of a title to filter the picker. Use `--continue` to
open the latest session for this agent and project, or `--session ID` to open a known session ID.
Resume requires an agent that advertises
`session/load`. The client stores session IDs and titles in its own SQLite database under the platform
state directory. Set `ISAAC_CLIENT_STATE_DIR` to use another directory. The picker prefers titles sent by
the agent and starts with a short title from the first prompt. When the agent advertises
`session/list`, the client also discovers sessions saved by that agent. The agent provides the
transcript on load.
With Isaac, this includes any checkpoint recovery notice.

For agents that support session loading, the client remembers the model selected with the picker or
confirmed by the agent's `/model ID` command. A fresh session uses that preference when the same
agent command still offers it. A resumed session uses the model restored by the agent.

## ACP support

The client uses ACP v1 over stdio with SDK `0.11.x`. It supports text prompts and streamed text and
thinking, tool updates and edit diffs, item plans, permission requests, cancellation, session loading,
session titles, context usage, advertised slash commands, session config options, form elicitation, and agent-directed
authentication. Use `--mcp-config PATH` to provide a JSON array of ACP MCP server descriptors for session
setup. Workspace tools run on the agent side with Isaac.

## Development

Run these checks from `client/`.

```bash
uv sync --locked
uv run --locked pytest
uv run --locked mypy
uv build --wheel
```
