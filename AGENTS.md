# AGENTS.md

This document provides instructions for agents working on the `isaac` codebase.

## Project Setup

This project uses `uv` for environment and project management.

- To install dependencies: `uv pip install -e .`
- To run the agent in ACP mode: `uv run isaac --acp`
- To run the agent in simple interactive mode: `uv run isaac`

## Linting and Formatting

This project uses `ruff` for linting and formatting.

- To format the code: `uv run ruff format .`
- To check for linting errors: `uv run ruff check .`

## Testing

This project uses `pytest` with `pytest-asyncio` for testing.

- To run tests: `uv run pytest`
