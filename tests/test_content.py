from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from acp import PromptRequest
from acp.helpers import embedded_text_resource, resource_block
from acp.schema import AgentMessageChunk

from tests.utils import make_function_agent


@pytest.mark.asyncio
async def test_content_blocks_use_embedded_resources():
    conn = AsyncMock()
    agent = make_function_agent(conn)

    block = resource_block(embedded_text_resource("uri:sample", "embedded text"))
    session_id = "content-session"
    response = await agent.prompt(PromptRequest(sessionId=session_id, prompt=[block]))

    assert conn.sessionUpdate.call_count >= 1
    notification = conn.sessionUpdate.call_args_list[-1][0][0]
    assert isinstance(notification.update, AgentMessageChunk)
    assert notification.update.content.text
    assert response.stopReason == "end_turn"
