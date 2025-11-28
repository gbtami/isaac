"""Terminal management mapped to ACP terminal endpoints.

See: https://agentclientprotocol.com/protocol/terminals
"""

from __future__ import annotations

import asyncio
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

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
from acp.schema import TerminalExitStatus

from .fs import resolve_path_for_session


@dataclass
class TerminalState:
    proc: asyncio.subprocess.Process
    output_limit: Optional[int] = None


async def _read_nonblocking(stream: asyncio.StreamReader | None, limit: int = 65536) -> str:
    if stream is None:
        return ""
    try:
        data = await asyncio.wait_for(stream.read(limit), timeout=0.01)
        return data.decode(errors="ignore") if data else ""
    except asyncio.TimeoutError:
        return ""


async def create_terminal(
    session_cwds: Dict[str, Path],
    terminals: Dict[str, TerminalState],
    params: CreateTerminalRequest,
) -> CreateTerminalResponse:
    import os
    import uuid

    terminal_id = str(uuid.uuid4())
    cwd = (
        resolve_path_for_session(session_cwds, params.sessionId, params.cwd)
        if params.cwd
        else Path.cwd()
    )

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
    terminals[terminal_id] = TerminalState(proc=proc, output_limit=params.outputByteLimit)
    return CreateTerminalResponse(terminalId=terminal_id)


async def terminal_output(
    terminals: Dict[str, TerminalState],
    params: TerminalOutputRequest,
) -> TerminalOutputResponse:
    state = terminals.get(params.terminalId)
    if not state:
        return TerminalOutputResponse(output="", truncated=False, exitStatus=None)

    stdout = await _read_nonblocking(state.proc.stdout)
    stderr = await _read_nonblocking(state.proc.stderr)
    combined = (stdout or "") + (stderr or "")

    truncated = False
    if state.output_limit is not None and len(combined.encode()) > state.output_limit:
        truncated = True
        combined = combined.encode()[: state.output_limit].decode(errors="ignore")

    exit_status = None
    if state.proc.returncode is not None:
        if state.proc.returncode < 0:
            sig = abs(state.proc.returncode)
            try:
                sig_name = signal.Signals(sig).name
            except Exception:
                sig_name = f"SIG{sig}"
            exit_status = TerminalExitStatus(exitCode=None, signal=sig_name)
        else:
            exit_status = TerminalExitStatus(exitCode=state.proc.returncode, signal=None)

    return TerminalOutputResponse(output=combined, truncated=truncated, exitStatus=exit_status)


async def wait_for_terminal_exit(
    terminals: Dict[str, TerminalState],
    params: WaitForTerminalExitRequest,
) -> WaitForTerminalExitResponse:
    state = terminals.get(params.terminalId)
    if not state:
        return WaitForTerminalExitResponse(exitCode=None, signal=None)
    returncode = await state.proc.wait()
    if returncode is None:
        return WaitForTerminalExitResponse(exitCode=None, signal=None)
    if returncode < 0:
        sig = abs(returncode)
        try:
            sig_name = signal.Signals(sig).name
        except Exception:
            sig_name = f"SIG{sig}"
        return WaitForTerminalExitResponse(exitCode=None, signal=sig_name)
    return WaitForTerminalExitResponse(exitCode=returncode, signal=None)


async def kill_terminal(
    terminals: Dict[str, TerminalState],
    params: KillTerminalCommandRequest,
) -> KillTerminalCommandResponse:
    state = terminals.get(params.terminalId)
    if state and state.proc.returncode is None:
        state.proc.kill()
    return KillTerminalCommandResponse()


async def release_terminal(
    terminals: Dict[str, TerminalState],
    params: ReleaseTerminalRequest,
) -> ReleaseTerminalResponse:
    state = terminals.pop(params.terminalId, None)
    if state and state.proc.returncode is None:
        state.proc.terminate()
    return ReleaseTerminalResponse()
