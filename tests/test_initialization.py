from __future__ import annotations

import pytest
from acp import PROTOCOL_VERSION, text_block
from acp.agent.connection import AgentSideConnection
from acp.schema import AgentMessageChunk
from unittest.mock import AsyncMock

from tests.utils import make_function_agent, make_error_agent


@pytest.mark.asyncio
async def test_initialize_includes_tools():
    conn = AsyncMock(spec=AgentSideConnection)
    agent = make_function_agent(conn)

    response = await agent.initialize(protocol_version=PROTOCOL_VERSION)

    assert response.protocol_version == PROTOCOL_VERSION
    assert response.agent_info.name == "isaac"


@pytest.mark.asyncio
async def test_prompt_echoes_plain_text():
    conn = AsyncMock(spec=AgentSideConnection)
    agent = make_function_agent(conn)

    session_id = "test-session"
    response = await agent.prompt(prompt=[text_block("hello world")], session_id=session_id)

    assert conn.session_update.call_count >= 1
    messages = [
        call.kwargs["update"]
        for call in conn.session_update.call_args_list
        if isinstance(call.kwargs.get("update"), AgentMessageChunk)
    ]
    assert messages, "Expected at least one AgentMessageChunk"
    assert any(getattr(m.content, "text", "") for m in messages)
    assert response.stop_reason == "end_turn"


@pytest.mark.asyncio
async def test_provider_error_is_sent_to_client():
    conn = AsyncMock(spec=AgentSideConnection)
    agent = make_error_agent(conn)

    session_id = "err-session"
    response = await agent.prompt(prompt=[text_block("do something")], session_id=session_id)

    assert response.stop_reason == "end_turn"
    updates = [
        call.kwargs["update"]
        for call in conn.session_update.call_args_list
        if isinstance(call.kwargs.get("update"), AgentMessageChunk)
    ]
    assert any("Model/provider error" in getattr(u.content, "text", "") for u in updates)
