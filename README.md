# 🍏 Isaac ACP 🍏 (`isaac-acp`)

[![PyPI version](https://img.shields.io/pypi/v/isaac-acp)](https://pypi.org/project/isaac-acp/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/isaac-acp)](https://pypi.org/project/isaac-acp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Lint](https://github.com/gbtami/isaac/actions/workflows/lint.yml/badge.svg)](https://github.com/gbtami/isaac/actions/workflows/lint.yml)
[![Tests](https://github.com/gbtami/isaac/actions/workflows/test.yml/badge.svg)](https://github.com/gbtami/isaac/actions/workflows/test.yml)

Isaac is an ACP-compliant coding agent and reference CLI client.

- Agent implementation: `isaac.agent`
- Client REPL implementation: `isaac.client`
- Protocol: [Agent Client Protocol (ACP)](https://agentclientprotocol.com/protocol/)

> Since Newton discovered gravity, everything's been going downhill.
## Installation

```bash
pip install isaac-acp
```

Optional Harness experiments are not installed by default:

```bash
pip install "isaac-acp[harness]"
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

1. `<platform config dir>/isaac/.env` (via `platformdirs`; Linux example: `~/.config/isaac/.env`)

Common variables:

- `OPENROUTER_API_KEY` (or provider-specific model keys)
- `ISAAC_ACP_STDIO_BUFFER_LIMIT_BYTES` (optional ACP stdio buffer override)

## Features

- ACP 0.10 session config options for mode/model selection
- Pydantic AI 2.x capability-based agent assembly
- Prompt turns, tool calls, filesystem and terminal ACP flows
- ACP-backed approval flow for `run_command` with an experimental Pydantic AI capability bridge
- Optional Pydantic AI Harness CodeMode experiments via `ISAAC_HARNESS_CODE_MODE=1` and the `harness` extra
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

Development note: `uv sync` installs the default `dev` dependency group, so local check commands such as `uv run pytest`, `uv run ruff check .`, and `uv run mypy` work after a normal sync. The `test` extra is kept for pip/CI compatibility.

### Optional Harness experiments

Install the optional Harness extra when experimenting with Pydantic AI Harness integrations:

```bash
uv sync --extra harness
```

Optional Harness tools are disabled by default. FileSystem and Shell experiments are exposed with prefixed `harness_*` tool names so they do not replace Isaac's ACP-compatible tools:

```bash
ISAAC_HARNESS_FILESYSTEM=1 uv run isaac
ISAAC_HARNESS_SHELL=1 uv run isaac
ISAAC_HARNESS_CODE_MODE=1 uv run isaac
```

Review approval and sandbox behavior before using Harness tools with mutating operations.
