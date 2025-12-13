"""Agent-side ACP terminal implementation.

Implements terminal endpoints per https://agentclientprotocol.com/protocol/terminals
so a client can launch and manage terminals on the agent host.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Dict

from acp import (
    CreateTerminalRequest,
    CreateTerminalResponse,
    KillTerminalCommandRequest,
    KillTerminalCommandResponse,
    ReleaseTerminalRequest,
    ReleaseTerminalResponse,
    TerminalOutputRequest,
    TerminalOutputResponse,
    WaitForTerminalExitRequest,
    WaitForTerminalExitResponse,
)

from isaac.agent.fs import resolve_path_for_session
from isaac.shared.terminal_common import TerminalState, build_exit_status, read_nonblocking


async def create_terminal(
    session_cwds: Dict[str, Path],
    terminals: Dict[str, TerminalState],
    params: CreateTerminalRequest,
) -> CreateTerminalResponse:
    """Spawn a terminal on the agent host (Terminals spec)."""
    import os
    import uuid

    terminal_id = str(uuid.uuid4())
    cwd = resolve_path_for_session(session_cwds, params.session_id, params.cwd) if params.cwd else Path.cwd()

    env = os.environ.copy()
    if params.env:
        for ev in params.env:
            env[ev.name] = ev.value

    cmd = [params.command, *(params.args or [])]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    terminals[terminal_id] = TerminalState(proc=proc, output_limit=params.output_byte_limit)
    return CreateTerminalResponse(terminal_id=terminal_id)


async def terminal_output(
    terminals: Dict[str, TerminalState],
    params: TerminalOutputRequest,
) -> TerminalOutputResponse:
    """Return incremental stdout/stderr for a terminal."""
    state = terminals.get(params.terminal_id)
    if not state:
        return TerminalOutputResponse(output="", truncated=False, exit_status=None)

    stdout = await read_nonblocking(state.proc.stdout)
    stderr = await read_nonblocking(state.proc.stderr)
    combined = (stdout or "") + (stderr or "")

    truncated = False
    if state.output_limit is not None and len(combined.encode()) > state.output_limit:
        truncated = True
        combined = combined.encode()[: state.output_limit].decode(errors="ignore")

    exit_status = build_exit_status(state.proc.returncode)

    return TerminalOutputResponse(output=combined, truncated=truncated, exit_status=exit_status)


async def wait_for_terminal_exit(
    terminals: Dict[str, TerminalState],
    params: WaitForTerminalExitRequest,
) -> WaitForTerminalExitResponse:
    """Await process completion for a terminal."""
    state = terminals.get(params.terminal_id)
    if not state:
        return WaitForTerminalExitResponse(exit_code=None, signal=None)
    returncode = await state.proc.wait()
    exit_status = build_exit_status(returncode)
    return WaitForTerminalExitResponse(
        exit_code=exit_status.exit_code if exit_status else None,
        signal=exit_status.signal if exit_status else None,
    )


async def kill_terminal(
    terminals: Dict[str, TerminalState],
    params: KillTerminalCommandRequest,
) -> KillTerminalCommandResponse:
    """Kill a running terminal process."""
    state = terminals.get(params.terminal_id)
    if state and state.proc.returncode is None:
        state.proc.kill()
    return KillTerminalCommandResponse()


async def release_terminal(
    terminals: Dict[str, TerminalState],
    params: ReleaseTerminalRequest,
) -> ReleaseTerminalResponse:
    """Release terminal resources, terminating if still alive."""
    state = terminals.pop(params.terminal_id, None)
    if state and state.proc.returncode is None:
        state.proc.terminate()
    return ReleaseTerminalResponse()
