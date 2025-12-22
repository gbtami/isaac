"""Interactive ACP client with REPL, tool calls, and terminal support."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import asyncio.subprocess as aio_subprocess
from acp import PROTOCOL_VERSION
from acp.core import connect_to_agent
from acp.schema import ClientCapabilities, FileSystemCapability, Implementation

from isaac.client.mcp_config import load_mcp_config
from isaac.client.acp_client import ACPClient
from isaac.client.repl import interactive_loop
from isaac.client.session_state import SessionUIState


async def run_client(program: str, args: Iterable[str], mcp_servers: list[Any]) -> int:
    _setup_client_logging()

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
    agent_caps = getattr(init_resp, "agent_capabilities", None)
    prompt_caps = getattr(agent_caps, "prompt_capabilities", None)
    state.agent_prompt_embedded_context = bool(getattr(prompt_caps, "embedded_context", False))
    meta = getattr(agent_caps, "field_meta", {}) or {}
    ext_methods = meta.get("extMethods", []) if isinstance(meta, dict) else []
    if isinstance(ext_methods, list):
        state.agent_ext_methods = {str(name) for name in ext_methods}

    state.mcp_servers = [
        (srv.get("name") if isinstance(srv, dict) else getattr(srv, "name", "<server>")) or "<server>"
        for srv in mcp_servers
    ]
    cwd = os.getcwd()
    session = await conn.new_session(cwd=cwd, mcp_servers=mcp_servers)
    state.session_id = session.session_id
    state.cwd = cwd
    if "model/list" in state.agent_ext_methods:
        try:
            ext_models = await conn.ext_method("model/list", {"session_id": session.session_id})
            if isinstance(ext_models, dict):
                current = ext_models.get("current")
                if isinstance(current, str):
                    state.current_model = current
                    state.notify_changed()
        except Exception:
            state.current_model = state.current_model
    if getattr(session, "modes", None) and getattr(session.modes, "current_mode_id", None):
        state.current_mode = session.modes.current_mode_id
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

    log_dir = Path.home() / ".isaac"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "acp_client.log"

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file)],
    )
