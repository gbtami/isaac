"""Tests for ACP agent functionality."""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, patch

from acp import (
    AgentSideConnection,
    InitializeRequest,
    InitializeResponse,
    PromptRequest,
    PromptResponse,
    session_notification,
    text_block,
    tool_result,
    ToolCall,
    ContentBlock,
)
from acp.schema import Tool, ToolParameter

from src.isaac.agent import ACPAgent, get_tools
from src.isaac.tools.list_directory import list_files
from src.isaac.tools.read_file import read_file

# Use pydantic_ai's TestModel for deterministic testing
from pydantic_ai import TestModel


def test_get_tools():
    """Test that get_tools returns proper tool definitions."""
    tools = get_tools()
    assert len(tools) == 2
    
    # Check list_files tool
    list_tool = next(t for t in tools if t.function == "tool_list_files")
    assert list_tool.description == "List files and directories recursively"
    assert isinstance(list_tool.parameters, ToolParameter)
    assert "directory" in list_tool.parameters.properties
    assert "recursive" in list_tool.parameters.properties

    # Check read_file tool
    read_tool = next(t for t in tools if t.function == "tool_read_file")
    assert read_tool.description == "Read a file with optional line range"
    assert isinstance(read_tool.parameters, ToolParameter)
    assert "file_path" in read_tool.parameters.properties
    assert "start_line" in read_tool.parameters.properties
    assert "num_lines" in read_tool.parameters.properties


def test_initialize():
    """Test that initialize returns proper capabilities with tools."""
    # Set up test
    conn = AsyncMock(spec=AgentSideConnection)
    agent = ACPAgent(conn)
    
    # Call initialize
    params = InitializeRequest()
    response = asyncio.run(agent.initialize(params))
    
    # Check response
    assert isinstance(response, InitializeResponse)
    assert response.protocolVersion == "0.1.0"  # Assuming this is the version
    assert response.agentInfo.name == "isaac"
    
    # Check capabilities
    capabilities = response.agentCapabilities
    assert hasattr(capabilities, "tools")
    assert len(capabilities.tools) == 2
    
    # Verify the tools match get_tools()
    expected_tools = get_tools()
    
    # Compare tool functions
    actual_functions = [t.function for t in capabilities.tools]
    expected_functions = [t.function for t in expected_tools]
    assert sorted(actual_functions) == sorted(expected_functions)    
    # Verify the tools match get_tools()
    expected_tools = get_tools()
    assert sorted(actual_functions) == sorted(expected_functions)


def test_handle_list_files_tool_call():
    """Test that the agent properly handles a list_files tool call."""
    # Set up test
    conn = AsyncMock(spec=AgentSideConnection)
    agent = ACPAgent(conn)
    
    # Create a prompt with a tool call
    tool_call_block = ContentBlock(
        toolCall=ToolCall(
            toolCallId="test-call-123",
            function="tool_list_files",
            arguments={"directory": ".", "recursive": True}
        )
    )
    params = PromptRequest(
        sessionId="test-session-456",
        prompt=[tool_call_block]
    )
    
    # Call prompt method
    response = asyncio.run(agent.prompt(params))
    
    # Check that sessionUpdate was called with tool result
    conn.sessionUpdate.assert_called_once()
    call_args = conn.sessionUpdate.call_args[0][0]  # First positional argument
    
    # Check it's a session notification
    assert hasattr(call_args, "sessionUpdate")
    update = call_args.sessionUpdate
    assert update.sessionId == "test-session-456"
    
    # Check it contains a tool_result
    assert hasattr(update, "tool_result")
    tool_result_block = update.tool_result
    assert tool_result_block.tool_call_id == "test-call-123"
    
    # The content should be a dict with 'content' and 'error'
    assert isinstance(tool_result_block.content, dict)
    assert 'content' in tool_result_block.content
    assert 'error' in tool_result_block.content
    
    # Check response
    assert isinstance(response, PromptResponse)
    assert response.stopReason == "tool_calls"


def test_handle_unknown_tool_call():
    """Test that the agent handles unknown tool functions gracefully."""
    # Set up test
    conn = AsyncMock(spec=AgentSideConnection)
    agent = ACPAgent(conn)
    
    # Create a prompt with an unknown tool call
    tool_call_block = ContentBlock(
        toolCall=ToolCall(
            toolCallId="test-call-789",
            function="unknown_tool",
            arguments={}
        )
    )
    params = PromptRequest(
        sessionId="test-session-000",
        prompt=[tool_call_block]
    )
    
    # Call prompt method
    response = asyncio.run(agent.prompt(params))
    
    # Check that sessionUpdate was called with error result
    conn.sessionUpdate.assert_called_once()
    call_args = conn.sessionUpdate.call_args[0][0]
    
    # Check it's a session notification
    assert hasattr(call_args, "sessionUpdate")
    update = call_args.sessionUpdate
    assert update.sessionId == "test-session-000"
    
    # Check it contains a tool_result
    assert hasattr(update, "tool_result")
    tool_result_block = update.tool_result
    assert tool_result_block.tool_call_id == "test-call-789"
    
    # The content should contain an error
    assert isinstance(tool_result_block.content, dict)
    assert 'error' in tool_result_block.content
    assert 'Unknown tool function' in tool_result_block.content['error']
    assert 'content' in tool_result_block.content
    assert tool_result_block.content['content'] is None
    
    # Check response
    assert isinstance(response, PromptResponse)
    assert response.stopReason == "tool_calls"


@pytest.mark.asyncio
async def test_fallback_to_ai():
    """Test that the agent falls back to AI processing when no tool calls are present."""
    # Use TestModel to make AI responses deterministic
    test_model = TestModel('test')
    test_model.add_response('Hello, world!')
    
    # Use patch to replace the pydantic_agent module variable
    with patch('src.isaac.agent.pydantic_agent') as mock_pydantic_agent:
        # Create a mock agent that uses our test model
        mock_agent = AsyncMock()
        mock_run = AsyncMock()
        mock_run.output = 'Hello, world!'
        mock_agent.run.return_value = mock_run
        mock_pydantic_agent = mock_agent

        # Set up test
        conn = AsyncMock(spec=AgentSideConnection)
        agent_instance = ACPAgent(conn)
        
        # Create a prompt with only text
        text_block = ContentBlock(text="Hello")
        params = PromptRequest(
            sessionId="test-session-111",
            prompt=[text_block]
        )
        
        # Call prompt method
        response = await agent_instance.prompt(params)
        
        # Check that sessionUpdate was called with AI response
        conn.sessionUpdate.assert_called_once()
        call_args = conn.sessionUpdate.call_args[0][0]
        
        # Check it's a session notification
        assert hasattr(call_args, "sessionUpdate")
        update = call_args.sessionUpdate
        assert update.sessionId == "test-session-111"
        
        # Check it contains an agent_message with text
        assert hasattr(update, "agent_message")
        message = update.agent_message
        assert hasattr(message, "content")
        assert message.content.text == "Hello, world!"
        
        # Check response
        assert isinstance(response, PromptResponse)
        assert response.stopReason == "end_turn"
    

def test_handle_read_file_tool_call():
    """Test that the agent properly handles a read_file tool call."""
    # Set up test
    conn = AsyncMock(spec=AgentSideConnection)
    agent = ACPAgent(conn)
    
    # Create a prompt with a tool call
    tool_call_block = ContentBlock(
        toolCall=ToolCall(
            toolCallId="test-call-456",
            function="tool_read_file",
            arguments={"file_path": __file__, "start_line": 1, "num_lines": 5}
        )
    )
    params = PromptRequest(
        sessionId="test-session-789",
        prompt=[tool_call_block]
    )
    
    # Call prompt method
    response = asyncio.run(agent.prompt(params))
    
    # Check that sessionUpdate was called with tool result
    conn.sessionUpdate.assert_called_once()
    call_args = conn.sessionUpdate.call_args[0][0]
    
    # Check it's a session notification
    assert hasattr(call_args, "sessionUpdate")
    update = call_args.sessionUpdate
    assert update.sessionId == "test-session-789"
    
    # Check it contains a tool_result
    assert hasattr(update, "tool_result")
    tool_result_block = update.tool_result
    assert tool_result_block.tool_call_id == "test-call-456"
    
    # The content should be a dict with 'content', 'num_tokens', 'error'
    assert isinstance(tool_result_block.content, dict)
    assert 'content' in tool_result_block.content
    assert 'num_tokens' in tool_result_block.content
    assert 'error' in tool_result_block.content
    
    # Check response
    assert isinstance(response, PromptResponse)
    assert response.stopReason == "tool_calls"


def test_handle_unknown_tool_call():
    """Test that the agent handles unknown tool functions gracefully."""
    # Set up test
    conn = AsyncMock(spec=AgentSideConnection)
    agent = ACPAgent(conn)
    
    # Create a prompt with an unknown tool call
    tool_call_block = ContentBlock(
        toolCall=ToolCall(
            toolCallId="test-call-789",
            function="unknown_tool",
            arguments={}
        )
    )
    params = PromptRequest(
        sessionId="test-session-000",
        prompt=[tool_call_block]
    )
    
    # Call prompt method
    response = asyncio.run(agent.prompt(params))
    
    # Check that sessionUpdate was called with error result
    conn.sessionUpdate.assert_called_once()
    call_args = conn.sessionUpdate.call_args[0][0]
    
    # Check it's a session notification
    assert hasattr(call_args, "sessionUpdate")
    update = call_args.sessionUpdate
    assert update.sessionId == "test-session-000"
    
    # Check it contains a tool_result
    assert hasattr(update, "tool_result")
    tool_result_block = update.tool_result
    assert tool_result_block.tool_call_id == "test-call-789"
    
    # The content should contain an error
    assert isinstance(tool_result_block.content, dict)
    assert 'error' in tool_result_block.content
    assert 'Unknown tool function' in tool_result_block.content['error']
    assert 'content' in tool_result_block.content
    assert tool_result_block.content['content'] is None
    
    # Check response
    assert isinstance(response, PromptResponse)
    assert response.stopReason == "tool_calls"


@pytest.mark.asyncio
async def test_fallback_to_ai():
    """Test that the agent falls back to AI processing when no tool calls are present."""
    # Use TestModel to make AI responses deterministic
    test_model = TestModel('test')
    test_model.add_response('Hello, world!')
    
    # Use patch to replace the pydantic_agent module variable
    with patch('src.isaac.agent.pydantic_agent') as mock_pydantic_agent:
        # Create a mock agent that uses our test model
        mock_agent = AsyncMock()
        mock_run = AsyncMock()
        mock_run.output = 'Hello, world!'
        mock_agent.run.return_value = mock_run
        mock_pydantic_agent = mock_agent

        # Set up test
        conn = AsyncMock(spec=AgentSideConnection)
        agent_instance = ACPAgent(conn)
        
        # Create a prompt with only text
        text_block = ContentBlock(text="Hello")
        params = PromptRequest(
            sessionId="test-session-111",
            prompt=[text_block]
        )
        
        # Call prompt method
        response = await agent_instance.prompt(params)
        
        # Check that sessionUpdate was called with AI response
        conn.sessionUpdate.assert_called_once()
        call_args = conn.sessionUpdate.call_args[0][0]
        
        # Check it's a session notification
        assert hasattr(call_args, "sessionUpdate")
        update = call_args.sessionUpdate
        assert update.sessionId == "test-session-111"
        
        # Check it contains an agent_message with text
        assert hasattr(update, "agent_message")
        message = update.agent_message
        assert hasattr(message, "content")
        assert message.content.text == "Hello, world!"
        
        # Check response
        assert isinstance(response, PromptResponse)
        assert response.stopReason == "end_turn"
