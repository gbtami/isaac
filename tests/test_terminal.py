from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from tests.utils import make_function_agent


@pytest.mark.asyncio
async def test_terminal_lifecycle(tmp_path: Path):
    conn = AsyncMock()
    agent = make_function_agent(conn)
    session = await agent.new_session(cwd=str(tmp_path), mcp_servers=[])

    # create a terminal that echoes once
    create_resp = await agent.create_terminal(
        command="echo",
        session_id=session.session_id,
        args=["hello"],
    )
    term_id = create_resp.terminal_id

    # fetch output
    out_resp = await agent.terminal_output(session_id=session.session_id, terminal_id=term_id)
    assert "hello" in out_resp.output

    # wait for exit
    exit_resp = await agent.wait_for_terminal_exit(
        session_id=session.session_id, terminal_id=term_id
    )
    assert exit_resp.exit_code == 0

    # create long-running and kill it
    create_resp2 = await agent.create_terminal(
        command="sleep", session_id=session.session_id, args=["5"]
    )
    term_id2 = create_resp2.terminal_id
    await agent.kill_terminal_command(session_id=session.session_id, terminal_id=term_id2)
    exit_resp2 = await agent.wait_for_terminal_exit(
        session_id=session.session_id, terminal_id=term_id2
    )
    assert exit_resp2.exit_code is not None or exit_resp2.signal is not None
