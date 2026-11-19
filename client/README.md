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

- `/client:model` opens a model selector when the agent advertises a model option.
- `/client:config` opens all advertised session config options, including grouped choices and toggles.
- `/client:plan` shows the latest plan. The status line shows plan progress, the active tool, and
  elapsed time. Reads and searches appear in grouped summaries. Failures and edit diffs appear inline.
- `/client:help` lists client controls and the agent's advertised slash commands.
- `/client:quit` exits. At the follow-up prompt, Enter queues another turn and Ctrl+C stops the
  current turn, clears the queue, and returns to the prompt when the agent acknowledges cancellation.

Permission requests show the supplied options and edit diffs when available. Select an option explicitly
in the terminal. Enter, EOF and scripted input cancel the request.

When an agent asks a structured question, enter the requested answers and review them before sending.
You can edit the answers or decline the form. Isaac offers these questions to clients that advertise
ACP form elicitation.

## Resume

Run the same agent command from the same project with `--resume` to open the named session picker.

```bash
isaac-client --resume --project-dir ~/playground isaac
```

Type part of a title to filter the picker, then choose a result by number. Use `--continue` to
open the latest session for this agent and project, or `--session ID` to open a known session ID.
Resume requires an agent that advertises
`session/load`. The client stores session IDs and titles in its own SQLite database under the platform
state directory. Set `ISAAC_CLIENT_STATE_DIR` to use another directory. The picker prefers titles sent by
the agent and starts with a short title from the first prompt. When the agent advertises
`session/list`, the client also discovers sessions saved by that agent. The agent provides the
transcript on load.
With Isaac, this includes any checkpoint recovery notice.

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
