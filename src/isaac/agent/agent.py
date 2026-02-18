"""App-level entrypoints and re-export of the core ACP agent implementation."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace

import logfire

from isaac.acp_runtime import ACP_STDIO_BUFFER_LIMIT_BYTES
from isaac.agent.acp_agent import ACPAgent
from isaac.agent.tools import run_tool
from isaac.log_utils import build_log_config, configure_logging, log_event

logger = logging.getLogger(__name__)

__all__ = ["ACPAgent", "run_acp_agent", "main_entry", "run_tool"]


async def run_acp_agent():
    """Run the ACP server."""
    from acp.core import run_agent  # Imported lazily to avoid hard dependency at import time

    _setup_logfire()
    _setup_acp_logging()
    log_event(logger, "agent.start", transport="stdio")

    await run_agent(ACPAgent(), stdio_buffer_limit_bytes=ACP_STDIO_BUFFER_LIMIT_BYTES)


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


def _setup_logfire() -> None:
    """Configure Logfire without writing to stdout (required for ACP stdio transport)."""

    logfire.configure(
        send_to_logfire="if-token-present",
        console=False,
        inspect_arguments=False,
    )
    logfire.instrument_pydantic_ai(
        include_binary_content=False,
        include_content=True,
        # Set include_content=False later if you want to stop exporting prompt/response text.
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
