"""Terminal handlers for ACP."""

from __future__ import annotations

from typing import Any

from acp import (
    CreateTerminalResponse,
    KillTerminalCommandResponse,
    ReleaseTerminalResponse,
    TerminalOutputResponse,
    WaitForTerminalExitResponse,
)
from acp.schema import (
    CreateTerminalRequest,
    KillTerminalCommandRequest,
    ReleaseTerminalRequest,
    TerminalOutputRequest,
    WaitForTerminalExitRequest,
)

from isaac.agent.agent_terminal import (
    create_terminal,
    kill_terminal,
    release_terminal,
    terminal_output,
    wait_for_terminal_exit,
)


class TerminalMixin:
    async def create_terminal(
        self,
        command: str,
        session_id: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: list[Any] | None = None,
        output_byte_limit: int | None = None,
        **kwargs: Any,
    ) -> CreateTerminalResponse:
        """Create a terminal on the agent host (Terminals section)."""
        params = CreateTerminalRequest(
            command=command,
            session_id=session_id,
            args=args,
            cwd=cwd,
            env=env,
            output_byte_limit=output_byte_limit,
            field_meta=kwargs or None,
        )
        return await create_terminal(self._session_cwds, self._terminals, params)

    async def terminal_output(self, session_id: str, terminal_id: str, **kwargs: Any) -> TerminalOutputResponse:
        """Stream terminal output (Terminals section)."""
        params = TerminalOutputRequest(session_id=session_id, terminal_id=terminal_id, field_meta=kwargs or None)
        return await terminal_output(self._terminals, params)

    async def wait_for_terminal_exit(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> WaitForTerminalExitResponse:
        """Wait for a terminal to exit (Terminals section)."""
        params = WaitForTerminalExitRequest(session_id=session_id, terminal_id=terminal_id, field_meta=kwargs or None)
        return await wait_for_terminal_exit(self._terminals, params)

    async def kill_terminal_command(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> KillTerminalCommandResponse:
        """Kill a running terminal command (Terminals section)."""
        params = KillTerminalCommandRequest(session_id=session_id, terminal_id=terminal_id, field_meta=kwargs or None)
        return await kill_terminal(self._terminals, params)

    async def release_terminal(self, session_id: str, terminal_id: str, **kwargs: Any) -> ReleaseTerminalResponse:
        """Release resources for a terminal (Terminals section)."""
        params = ReleaseTerminalRequest(session_id=session_id, terminal_id=terminal_id, field_meta=kwargs or None)
        return await release_terminal(self._terminals, params)
