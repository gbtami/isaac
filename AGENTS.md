# AGENTS.md

## Project Overview
isaac is an AI coding agent implementing the Agent Client Protocol and can be used with an ACP Client.
It is built with:
- **logic**: pydantic-ai https://github.com/pydantic/pydantic-ai
- **communication**: agent client protocol https://agentclientprotocol.com/overview/introduction via https://github.com/agentclientprotocol/python-sdk

## Project Setup

This project uses `uv` for environment and project management.

- To install dependencies: `uv pip install -e .`
- To run the client with the agent: `uv run python -m isaac.client uv run isaac`

## Linting and Formatting

This project uses `ruff` for linting and formatting.

- To format the code: `uv run ruff format .`
- To check for linting errors: `uv run ruff check .`

## Testing

This project uses `pytest` with `pytest-asyncio` for testing.

- To run tests: `uv run pytest`
