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
    updates = [call.kwargs["update"] for call in conn.session_update.call_args_list]  # type: ignore[attr-defined]
    agent_chunks = [u for u in updates if isinstance(u, AgentMessageChunk)]
    assert agent_chunks
    assert any(getattr(c.content, "text", "") for c in agent_chunks)
    assert response.stop_reason == "end_turn"
