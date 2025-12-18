"""Agent-side terminal helpers (decoupled from client package)."""

from __future__ import annotations

import asyncio
import signal
from dataclasses import dataclass
from typing import Optional

from acp.schema import TerminalExitStatus


@dataclass
class TerminalState:
    proc: asyncio.subprocess.Process
    output_limit: Optional[int] = None


async def read_nonblocking(
    stream: asyncio.StreamReader | None,
    limit: int = 65536,
) -> str:
    """Read without blocking to keep terminals responsive."""
    if stream is None:
        return ""
    try:
        data = await asyncio.wait_for(stream.read(limit), timeout=0.01)
        return data.decode(errors="ignore") if data else ""
    except asyncio.TimeoutError:
        return ""


def build_exit_status(returncode: int | None) -> TerminalExitStatus | None:
    if returncode is None:
        return None
    if returncode < 0:
        sig = abs(returncode)
        try:
            sig_name = signal.Signals(sig).name
        except Exception:
            sig_name = f"SIG{sig}"
        return TerminalExitStatus(exit_code=None, signal=sig_name)
    return TerminalExitStatus(exit_code=returncode, signal=None)
