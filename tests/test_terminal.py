from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from acp import (
    CreateTerminalRequest,
    KillTerminalCommandRequest,
    NewSessionRequest,
    TerminalOutputRequest,
    WaitForTerminalExitRequest,
)

from tests.utils import make_function_agent


@pytest.mark.asyncio
async def test_terminal_lifecycle(tmp_path: Path):
    conn = AsyncMock()
    agent = make_function_agent(conn)
    session = await agent.newSession(NewSessionRequest(cwd=str(tmp_path), mcpServers=[]))

    # create a terminal that echoes once
    create_resp = await agent.createTerminal(
        CreateTerminalRequest(sessionId=session.sessionId, command="echo", args=["hello"])
    )
    term_id = create_resp.terminalId

    # fetch output
    out_resp = await agent.terminalOutput(
        TerminalOutputRequest(sessionId=session.sessionId, terminalId=term_id)
    )
    assert "hello" in out_resp.output

    # wait for exit
    exit_resp = await agent.waitForTerminalExit(
        WaitForTerminalExitRequest(sessionId=session.sessionId, terminalId=term_id)
    )
    assert exit_resp.exitCode == 0

    # create long-running and kill it
    create_resp2 = await agent.createTerminal(
        CreateTerminalRequest(sessionId=session.sessionId, command="sleep", args=["5"])
    )
    term_id2 = create_resp2.terminalId
    await agent.killTerminalCommand(
        KillTerminalCommandRequest(sessionId=session.sessionId, terminalId=term_id2)
    )
    exit_resp2 = await agent.waitForTerminalExit(
        WaitForTerminalExitRequest(sessionId=session.sessionId, terminalId=term_id2)
    )
    assert exit_resp2.exitCode is not None or exit_resp2.signal is not None
