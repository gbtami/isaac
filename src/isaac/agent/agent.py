"""App-level entrypoints and re-export of the core ACP agent implementation."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace

from isaac.acp_compat import enable_session_config_options_api
from isaac.agent.acp_agent import ACPAgent
from isaac.agent.tools import run_tool
from isaac.log_utils import build_log_config, configure_logging, log_event

logger = logging.getLogger(__name__)

__all__ = ["ACPAgent", "run_acp_agent", "main_entry", "run_tool"]


async def run_acp_agent():
    """Run the ACP server."""
    from acp.core import run_agent  # Imported lazily to avoid hard dependency at import time

    enable_session_config_options_api()
    _setup_acp_logging()
    log_event(logger, "agent.start", transport="stdio")

    await run_agent(ACPAgent())


def _setup_acp_logging():
    config = build_log_config(log_file_name="isaac.log")
    configure_logging(
        replace(
            config,
            logger_levels={
                "isaac": config.level,
                "acp": config.level,
                "pydantic_ai": config.level,
                "pydantic_ai.providers": config.level,
            },
        )
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
