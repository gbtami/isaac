from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from acp.agent.connection import AgentSideConnection
from acp import text_block
from acp.schema import AgentMessageChunk

from tests.utils import make_function_agent


@pytest.mark.asyncio
async def test_set_session_model_changes_runner():
    conn = AsyncMock(spec=AgentSideConnection)
    agent = make_function_agent(conn)
    session = await agent.new_session(cwd="/", mcp_servers=[])

    await agent.set_session_model(model_id="function-model", session_id=session.session_id)

    response = await agent.prompt(prompt=[text_block("hello")], session_id=session.session_id)

    conn.session_update.assert_called()
    notification = conn.session_update.call_args.kwargs["update"]
    assert isinstance(notification, AgentMessageChunk)
    assert notification.content.text
    assert "Error" not in notification.content.text
    assert response.stop_reason == "end_turn"
