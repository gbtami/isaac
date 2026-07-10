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

Install Isaac as a `uv` tool:

```bash
uv tool install isaac-acp
```

Optional Harness experiments are not installed by default:

```bash
uv tool install "isaac-acp[harness]" --force
```

## Quickstart

Run the agent:

```bash
isaac
```

Run the bundled client against the bundled agent without creating a project checkout:

```bash
uv tool run --from isaac-acp python -m isaac.client isaac
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
- `ISAAC_SHELL_ALLOWLIST` (optional comma/newline-separated regex allowlist for `run_command`)
- `ISAAC_SHELL_DENYLIST` (optional comma/newline-separated regex denylist for `run_command`)
- `ISAAC_COMMAND_ENV_DENYLIST` (optional comma/newline-separated regexes for additional environment variable names to remove from command subprocesses)

## Features

- ACP 0.11 session config options for mode/model selection
- Pydantic AI 2.x capability-based agent assembly
- Prompt turns, tool calls, filesystem and terminal ACP flows
- Workspace-contained filesystem tools with binary-file guards, protected write paths, and optional SHA-256 write preconditions
- ACP-backed approval flow for `run_command` with an experimental Pydantic AI capability bridge
- Optional Pydantic AI Harness CodeMode experiments via `ISAAC_HARNESS_CODE_MODE=1` and the `harness` extra
- Interactive client slash commands (`/mode`, `/model`, `/status`, `/usage`)
- MCP server config forwarding from the client to the agent

### Command environment security

`run_command` inherits the normal development environment but removes variables
whose names look secret-bearing, including names containing `TOKEN`, `SECRET`,
`PASSWORD`, `PRIVATE`, `CREDENTIAL`, `AUTH`, `COOKIE`, `API_KEY`, or
`ACCESS_KEY`. Matching is case-insensitive and substring-based. For example,
`SSH_AUTH_SOCK` is removed because its name contains `AUTH`, so commands run by
Isaac do not automatically inherit the caller's SSH agent.

`ISAAC_COMMAND_ENV_DENYLIST` can add more variable-name regular expressions to
remove. It cannot restore names removed by the built-in policy. Run authenticated
Git operations outside agent-executed commands, or provide only narrowly scoped
credentials through an explicitly reviewed mechanism.

These checks reduce accidental credential exposure, but they do not make
`run_command` a sandbox.

## Development

```bash
uv sync
uv run ruff format .
uv run ruff check .
uv run mypy src tests
uv run pytest
uv build --wheel --sdist
```

Development note: `uv sync` installs the default `dev` dependency group, so local check commands such as `uv run pytest`, `uv run ruff check .`, and `uv run mypy` work after a normal sync. The `test` extra is kept for packaging and CI compatibility.

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
