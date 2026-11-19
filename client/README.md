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
uv run --project client isaac-client --project-dir ~/playground uv run --project . isaac
```

Client options go before the agent command. The remaining arguments go to the agent.
The agent process starts in the caller's directory, and ACP supplies the project directory for the session.
Configure model credentials through the chosen agent. Isaac supports its own `isaac login` command.

## Terminal controls

- `/client:model` opens a model selector when the agent advertises a model option.
- `/client:config` opens all advertised session config options, including grouped choices and toggles.
- `/client:plan` shows the latest plan. Plan updates, tool activity and diffs also appear during a turn.
- `/client:help` lists client controls and the agent's advertised slash commands.
- `/client:quit` exits. Ctrl+C during a turn sends ACP cancellation before exiting.

Permission requests show the supplied options and edit diffs when available. Select an option explicitly
in the terminal. Enter, EOF and scripted input cancel the request.

## Resume

Run the same agent command from the same project with `--resume` to open the named session picker.

```bash
isaac-client --resume --project-dir ~/playground isaac
```

Use `--session ID` to open a known ACP session ID directly. Resume requires an agent that advertises
`session/load`. The client stores session IDs and titles in its own SQLite database under the platform
state directory. Set `ISAAC_CLIENT_STATE_DIR` to use another directory. The picker prefers titles sent by
the agent and starts with a short title from the first prompt. The agent provides the transcript on load.
With Isaac, this includes any checkpoint recovery notice.

## ACP support

The client uses ACP v1 over stdio with SDK `0.11.x`. It supports text prompts and streamed text and
thinking, tool updates and edit diffs, item plans, permission requests, cancellation, session loading,
session titles, context usage, advertised slash commands, session config options, and agent-directed
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
