"""App-level entrypoints and re-export of the core ACP agent implementation."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from isaac.agent.acp_agent import ACPAgent
from isaac.agent.tools import run_tool

__all__ = ["ACPAgent", "run_acp_agent", "main_entry", "run_tool"]


async def run_acp_agent():
    """Run the ACP server."""
    from acp.core import run_agent  # Imported lazily to avoid hard dependency at import time

    _setup_acp_logging()
    logging.getLogger("acp_server").info("Starting ACP server on stdio")

    await run_agent(ACPAgent())


def _setup_acp_logging():
    log_dir = Path.home() / ".isaac"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "acp_server.log"

    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file)],
    )


async def main(argv: list[str] | None = None):
    """Default entrypoint launches the ACP server on stdio."""
    return await run_acp_agent()


def main_entry():
    try:
        return asyncio.run(main())
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    main_entry()
