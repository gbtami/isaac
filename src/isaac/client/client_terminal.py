"""Client-side ACP terminal implementation.

Implements the client terminal methods per https://agentclientprotocol.com/protocol/terminals
so the agent can request terminals that run on the client's host.
"""

from __future__ import annotations

import asyncio.subprocess as aio_subprocess
import os
import uuid
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

from isaac.client.terminal_common import TerminalState, build_exit_status, read_nonblocking


class ClientTerminalManager:
    """Manage client-hosted terminals to satisfy ACP terminal requests."""

    def __init__(self, terminals: Dict[str, TerminalState] | None = None) -> None:
        self._terminals: Dict[str, TerminalState] = terminals or {}

    async def create_terminal(self, params: CreateTerminalRequest) -> CreateTerminalResponse:
        """Create a client-hosted terminal for agent use (Terminals spec)."""
        cwd = Path(params.cwd) if params.cwd else Path.cwd()
        env = os.environ.copy()
        if params.env:
            for ev in params.env:
                env[ev.name] = ev.value

        proc = await aio_subprocess.create_subprocess_exec(
            params.command,
            *(params.args or []),
            cwd=str(cwd),
            stdout=aio_subprocess.PIPE,
            stderr=aio_subprocess.PIPE,
            env=env,
        )

        terminal_id = str(uuid.uuid4())
        self._terminals[terminal_id] = TerminalState(proc=proc, output_limit=params.output_byte_limit)
        return CreateTerminalResponse(terminal_id=terminal_id)

    async def terminal_output(self, params: TerminalOutputRequest) -> TerminalOutputResponse:
        """Return streamed output for a client terminal."""
        state = self._terminals.get(params.terminal_id)
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

    async def wait_for_terminal_exit(self, params: WaitForTerminalExitRequest) -> WaitForTerminalExitResponse:
        """Wait for a terminal to exit."""
        state = self._terminals.get(params.terminal_id)
        if not state:
            return WaitForTerminalExitResponse(exit_code=None, signal=None)
        returncode = await state.proc.wait()
        exit_status = build_exit_status(returncode)
        return WaitForTerminalExitResponse(
            exit_code=exit_status.exit_code if exit_status else None,
            signal=exit_status.signal if exit_status else None,
        )

    async def kill_terminal(self, params: KillTerminalCommandRequest) -> KillTerminalCommandResponse:
        """Forcefully kill a terminal process."""
        state = self._terminals.get(params.terminal_id)
        if state and state.proc.returncode is None:
            state.proc.kill()
        return KillTerminalCommandResponse()

    async def release_terminal(self, params: ReleaseTerminalRequest) -> ReleaseTerminalResponse:
        """Release and terminate a client terminal if still running."""
        state = self._terminals.pop(params.terminal_id, None)
        if state and state.proc.returncode is None:
            state.proc.terminate()
        return ReleaseTerminalResponse()
