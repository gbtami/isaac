from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from acp.helpers import embedded_text_resource, resource_block
from acp.schema import AgentMessageChunk

from tests.utils import make_function_agent


@pytest.mark.asyncio
async def test_content_blocks_use_embedded_resources():
    conn = AsyncMock()
    agent = make_function_agent(conn)

    block = resource_block(embedded_text_resource("uri:sample", "embedded text"))
    session_id = "content-session"
    response = await agent.prompt(prompt=[block], session_id=session_id)

    assert conn.session_update.call_count >= 1
    notification = conn.session_update.call_args_list[-1].kwargs["update"]
    assert isinstance(notification, AgentMessageChunk)
    assert notification.content.text
    assert response.stop_reason == "end_turn"
