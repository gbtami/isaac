import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from acp import (
    AgentSideConnection,
    InitializeRequest,
    NewSessionRequest,
    PromptRequest,
    text_block,
    PROTOCOL_VERSION,
)

from isaac.main import ACPAgent, PydanticAgent, OpenAIChatModel, OpenAIProvider

# Mock PydanticAgent and its dependencies
@pytest.fixture
def mock_pydantic_agent():
    with patch("isaac.main.PydanticAgent", autospec=True) as mock_agent:
        yield mock_agent

@pytest.fixture
def mock_openai_provider():
    with patch("isaac.main.OpenAIProvider", autospec=True) as mock_provider:
        yield mock_provider

@pytest.fixture
def mock_openai_chat_model():
    with patch("isaac.main.OpenAIChatModel", autospec=True) as mock_model:
        yield mock_model

# Test ACPAgent methods
@pytest.mark.asyncio
async def test_acp_agent_initialize():
    """Test the initialize method of ACPAgent."""
    conn = AsyncMock(spec=AgentSideConnection)
    pydantic_agent = MagicMock(spec=PydanticAgent)
    agent = ACPAgent(conn, pydantic_agent)

    request = InitializeRequest(protocolVersion=PROTOCOL_VERSION)
    response = await agent.initialize(request)

    assert response.protocolVersion == PROTOCOL_VERSION
    assert response.agentInfo.name == "example-agent"

@pytest.mark.asyncio
async def test_acp_agent_new_session():
    """Test the newSession method of ACPAgent."""
    conn = AsyncMock(spec=AgentSideConnection)
    pydantic_agent = MagicMock(spec=PydanticAgent)
    agent = ACPAgent(conn, pydantic_agent)

    request = NewSessionRequest(cwd="/", mcpServers=[])
    response = await agent.newSession(request)

    assert response.sessionId is not None
    assert response.sessionId in agent._sessions

@pytest.mark.asyncio
async def test_acp_agent_prompt():
    """Test the prompt method of ACPAgent."""
    conn = AsyncMock(spec=AgentSideConnection)
    pydantic_agent = MagicMock(spec=PydanticAgent)

    # Mock the run method of the PydanticAgent
    mock_response = MagicMock()
    mock_response.output = "Test response"
    pydantic_agent.run = AsyncMock(return_value=mock_response)

    agent = ACPAgent(conn, pydantic_agent)
    session_id = "test-session"
    agent._sessions.add(session_id)

    request = PromptRequest(
        sessionId=session_id, prompt=[text_block("Hello")],
    )

    response = await agent.prompt(request)

    # Verify that the agent's run method was called with the correct prompt
    pydantic_agent.run.assert_called_once_with("Hello")

    # Verify that the sessionUpdate method was called with the correct response
    conn.sessionUpdate.assert_called_once()

    # Verify the prompt response
    assert response.stopReason == "end_turn"

@pytest.mark.asyncio
async def test_acp_agent_consecutive_prompts():
    """Test that the agent can handle two consecutive prompts."""
    conn = AsyncMock(spec=AgentSideConnection)
    pydantic_agent = MagicMock(spec=PydanticAgent)

    # Mock the run method to return different outputs for different inputs
    async def mock_run(prompt_text):
        mock_response = MagicMock()
        if "First" in prompt_text:
            mock_response.output = "Response to first"
        elif "Second" in prompt_text:
            mock_response.output = "Response to second"
        else:
            mock_response.output = "Default response"
        return mock_response

    pydantic_agent.run = AsyncMock(side_effect=mock_run)

    agent = ACPAgent(conn, pydantic_agent)
    session_id = "test-session-consecutive"
    agent._sessions.add(session_id)

    # First prompt
    request1 = PromptRequest(
        sessionId=session_id, prompt=[text_block("First prompt")],
    )
    response1 = await agent.prompt(request1)

    # Second prompt
    request2 = PromptRequest(
        sessionId=session_id, prompt=[text_block("Second prompt")],
    )
    response2 = await agent.prompt(request2)

    # Verify run method calls
    assert pydantic_agent.run.call_count == 2
    pydantic_agent.run.assert_any_call("First prompt")
    pydantic_agent.run.assert_any_call("Second prompt")

    # Verify sessionUpdate calls
    assert conn.sessionUpdate.call_count == 2

    # Verify prompt responses
    assert response1.stopReason == "end_turn"
    assert response2.stopReason == "end_turn"
