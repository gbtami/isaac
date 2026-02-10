# 🍏 Isaac ACP🍏 (`isaac-acp`)

[![PyPI version](https://img.shields.io/pypi/v/isaac-acp)](https://pypi.org/project/isaac-acp/)

Isaac is an ACP-compliant coding agent and reference CLI client.

- Agent implementation: `isaac.agent`
- Client REPL implementation: `isaac.client`
- Protocol: [Agent Client Protocol (ACP)](https://agentclientprotocol.com/protocol/)

> Since Newton discovered gravity, everything's been going downhill.
## Installation

```bash
pip install isaac-acp
```

## Quickstart

Run the agent:

```bash
isaac
```

Run the bundled client against the bundled agent:

```bash
python -m isaac.client isaac
```

Or when developing from source:

```bash
uv run python -m isaac.client uv run isaac
```

## Configuration

Isaac loads environment variables from:

1. `~/.config/isaac/.env`
2. `./.env` (current working directory, overrides shared values)

Common variables:

- `OPENROUTER_API_KEY` (or provider-specific model keys)
- `ISAAC_ACP_STDIO_BUFFER_LIMIT_BYTES` (optional ACP stdio buffer override)

## Features

- ACP 0.8 session config options for mode/model selection
- Prompt turns, tool calls, filesystem and terminal ACP flows
- Interactive client slash commands (`/mode`, `/model`, `/status`, `/usage`)
- MCP server config forwarding from the client to the agent

## Development

```bash
uv pip install -e .
uv run ruff format .
uv run ruff check .
uv run mypy src tests
uv run pytest
uv build --wheel --sdist
```
