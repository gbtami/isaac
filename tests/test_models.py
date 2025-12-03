from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from acp import AgentSideConnection, PromptRequest, SetSessionModelRequest, text_block, NewSessionRequest
from acp.schema import AgentMessageChunk

from tests.utils import make_function_agent


@pytest.mark.asyncio
async def test_set_session_model_changes_runner():
    conn = AsyncMock(spec=AgentSideConnection)
    agent = make_function_agent(conn)
    session = await agent.newSession(NewSessionRequest(cwd="/", mcpServers=[]))

    await agent.setSessionModel(
        SetSessionModelRequest(sessionId=session.sessionId, modelId="function-model")
    )

    response = await agent.prompt(
        PromptRequest(sessionId=session.sessionId, prompt=[text_block("hello")])
    )

    conn.sessionUpdate.assert_called()
    notification = conn.sessionUpdate.call_args[0][0]
    assert isinstance(notification.update, AgentMessageChunk)
    assert notification.update.content.text
    assert "Error" not in notification.update.content.text
    assert response.stopReason == "end_turn"
