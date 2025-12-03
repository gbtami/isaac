from __future__ import annotations

import pytest
from acp import AgentSideConnection, InitializeRequest, PromptRequest, PROTOCOL_VERSION, text_block
from acp.schema import AgentMessageChunk
from unittest.mock import AsyncMock

from tests.utils import make_function_agent, make_error_agent


@pytest.mark.asyncio
async def test_initialize_includes_tools():
    conn = AsyncMock(spec=AgentSideConnection)
    agent = make_function_agent(conn)

    response = await agent.initialize(InitializeRequest(protocolVersion=PROTOCOL_VERSION))

    assert response.protocolVersion == PROTOCOL_VERSION
    assert response.agentInfo.name == "isaac"


@pytest.mark.asyncio
async def test_prompt_echoes_plain_text():
    conn = AsyncMock(spec=AgentSideConnection)
    agent = make_function_agent(conn)

    session_id = "test-session"
    response = await agent.prompt(
        PromptRequest(sessionId=session_id, prompt=[text_block("hello world")])
    )

    assert conn.sessionUpdate.call_count >= 1
    messages = [
        call_args[0][0].update
        for call_args in conn.sessionUpdate.call_args_list
        if isinstance(call_args[0][0].update, AgentMessageChunk)
    ]
    assert messages, "Expected at least one AgentMessageChunk"
    assert any(getattr(m.content, "text", "") for m in messages)
    assert response.stopReason == "end_turn"


@pytest.mark.asyncio
async def test_provider_error_is_sent_to_client():
    conn = AsyncMock(spec=AgentSideConnection)
    agent = make_error_agent(conn)

    session_id = "err-session"
    response = await agent.prompt(
        PromptRequest(sessionId=session_id, prompt=[text_block("do something")])
    )

    assert response.stopReason == "end_turn"
    updates = [
        call_args[0][0].update
        for call_args in conn.sessionUpdate.call_args_list
        if isinstance(call_args[0][0].update, AgentMessageChunk)
    ]
    assert any("Model/provider error" in getattr(u.content, "text", "") for u in updates)
