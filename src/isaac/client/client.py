"""Interactive ACP client with REPL, tool calls, and terminal support."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import asyncio.subprocess as aio_subprocess
from acp import PROTOCOL_VERSION
from acp.core import connect_to_agent
from acp.schema import ClientCapabilities, FileSystemCapability, Implementation

from isaac.acp_compat import enable_session_config_options_api
from isaac.acp_runtime import ACP_STDIO_BUFFER_LIMIT_BYTES
from isaac.client.mcp_config import load_mcp_config
from isaac.client.acp_client import ACPClient, apply_session_config_options
from isaac.client.repl import interactive_loop
from isaac.client.session_state import SessionUIState
from isaac.log_utils import build_log_config, configure_logging, log_event

logger = logging.getLogger(__name__)


async def run_client(program: str, args: Iterable[str], mcp_servers: list[Any]) -> int:
    enable_session_config_options_api()
    _setup_client_logging()
    log_event(logger, "client.start", program=program)

    program_path = Path(program)
    spawn_program = program
    spawn_args = list(args)

    if program_path.exists() and not os.access(program_path, os.X_OK):
        spawn_program = sys.executable
        spawn_args = [str(program_path), *spawn_args]

    proc = await asyncio.create_subprocess_exec(
        spawn_program,
        *spawn_args,
        stdin=aio_subprocess.PIPE,
        stdout=aio_subprocess.PIPE,
        limit=ACP_STDIO_BUFFER_LIMIT_BYTES,
    )

    if proc.stdin is None or proc.stdout is None:
        print("Agent process does not expose stdio pipes", file=sys.stderr)
        return 1

    state = SessionUIState(current_mode="ask", current_model="unknown", mcp_servers=[])

    client_impl = ACPClient(state)
    conn = connect_to_agent(client_impl, proc.stdin, proc.stdout)

    init_resp = await conn.initialize(
        protocol_version=PROTOCOL_VERSION,
        client_capabilities=ClientCapabilities(
            fs=FileSystemCapability(read_text_file=False, write_text_file=False),
            terminal=True,
        ),
        client_info=Implementation(name="example-client", title="Example Client", version="0.1.0"),
    )
    if init_resp.protocol_version != PROTOCOL_VERSION:
        print(
            f"Incompatible ACP protocol version from agent: {init_resp.protocol_version}",
            file=sys.stderr,
        )
        with contextlib.suppress(Exception):
            await conn.close()
        if proc.returncode is None:
            proc.terminate()
            with contextlib.suppress(ProcessLookupError):
                await proc.wait()
        return 1
    state.mcp_servers = [
        (srv.get("name") if isinstance(srv, dict) else getattr(srv, "name", "<server>")) or "<server>"
        for srv in mcp_servers
    ]
    cwd = os.getcwd()
    session = await conn.new_session(cwd=cwd, mcp_servers=mcp_servers)
    state.session_id = session.session_id
    state.cwd = cwd
    if getattr(session, "config_options", None):
        apply_session_config_options(state, session.config_options or [])
    if getattr(session, "modes", None) and getattr(session.modes, "current_mode_id", None):
        state.current_mode = session.modes.current_mode_id
        state.notify_changed()
    if getattr(session, "models", None) and getattr(session.models, "current_model_id", None):
        state.current_model = session.models.current_model_id
        state.notify_changed()

    try:
        await interactive_loop(conn, session.session_id, state)
        return 0
    except KeyboardInterrupt:
        return 130
    finally:
        if proc.returncode is None:
            proc.terminate()
            with contextlib.suppress(ProcessLookupError):
                await proc.wait()


async def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Run ACP client against an agent program.")
    parser.add_argument(
        "--mcp-config",
        type=str,
        help="Path to JSON file containing ACP mcpServers array (stdio/http/sse entries).",
    )
    parser.add_argument("agent_program", help="Path to the agent program to launch")
    parser.add_argument("agent_args", nargs=argparse.REMAINDER, help="Arguments for the agent")
    args = parser.parse_args(argv[1:])

    mcp_servers: list[Any] = []
    if args.mcp_config:
        mcp_servers = load_mcp_config(args.mcp_config)

    return await run_client(args.agent_program, args.agent_args, mcp_servers)


def _setup_client_logging() -> None:
    """Initialize client logging to a file (mirrors agent logging convention)."""

    config = build_log_config(log_file_name="acp_client.log")
    configure_logging(
        replace(
            config,
            logger_levels={
                "isaac": config.level,
                "acp": config.level,
            },
        )
    )
