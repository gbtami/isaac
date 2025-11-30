from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from acp import AgentSideConnection, InitializeRequest, PromptRequest, PROTOCOL_VERSION, text_block
from acp.schema import AgentMessageChunk, ToolCallProgress, ToolCallStart
from acp import SetSessionModelRequest
from acp.helpers import embedded_text_resource, resource_block
from pydantic_ai import Agent as PydanticAgent  # type: ignore
from pydantic_ai.models.test import TestModel  # type: ignore

from isaac.agent import ACPAgent
from acp import NewSessionRequest, ReadTextFileRequest, WriteTextFileRequest
from acp import (
    CreateTerminalRequest,
    TerminalOutputRequest,
    WaitForTerminalExitRequest,
    KillTerminalCommandRequest,
)


def make_function_agent(conn: AgentSideConnection) -> ACPAgent:
    """Helper to build ACPAgent with a deterministic in-process model."""
    return ACPAgent(conn, ai_runner=PydanticAgent(TestModel(call_tools=[])))


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
async def test_tool_list_files_sends_progress(tmp_path: Path):
    (tmp_path / "file_a.txt").write_text("data")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "file_b.txt").write_text("more")

    conn = AsyncMock(spec=AgentSideConnection)
    agent = make_function_agent(conn)

    session_id = "tool-session"
    response = await agent.prompt(
        PromptRequest(sessionId=session_id, prompt=[text_block(f"tool:list_files {tmp_path}")])
    )

    assert response.stopReason == "end_turn"
    assert conn.sessionUpdate.call_count == 2

    start_call = conn.sessionUpdate.call_args_list[0][0][0]
    update_call = conn.sessionUpdate.call_args_list[1][0][0]

    assert isinstance(start_call.update, ToolCallStart)
    assert start_call.update.status == "in_progress"
    assert start_call.sessionId == session_id

    assert isinstance(update_call.update, ToolCallProgress)
    assert update_call.update.status == "completed"
    assert update_call.update.rawOutput["error"] is None


@pytest.mark.asyncio
async def test_tool_read_file_returns_content(tmp_path: Path):
    target = tmp_path / "sample.txt"
    target.write_text("line1\nline2\nline3\n")

    conn = AsyncMock(spec=AgentSideConnection)
    agent = make_function_agent(conn)

    session_id = "read-session"
    response = await agent.prompt(
        PromptRequest(sessionId=session_id, prompt=[text_block(f"tool:read_file {target} 2 1")])
    )

    assert response.stopReason == "end_turn"
    assert conn.sessionUpdate.call_count == 2

    update_call = conn.sessionUpdate.call_args_list[1][0][0]
    assert isinstance(update_call.update, ToolCallProgress)
    assert update_call.update.status == "completed"
    assert "line2" in update_call.update.rawOutput["content"]


@pytest.mark.asyncio
async def test_tool_run_command_executes(tmp_path: Path):
    conn = AsyncMock(spec=AgentSideConnection)
    agent = make_function_agent(conn)

    session_id = "cmd-session"
    response = await agent.prompt(
        PromptRequest(sessionId=session_id, prompt=[text_block("tool:run_command echo hello")])
    )

    assert response.stopReason == "end_turn"
    assert conn.sessionUpdate.call_count == 2

    update_call = conn.sessionUpdate.call_args_list[1][0][0]
    assert isinstance(update_call.update, ToolCallProgress)
    assert update_call.update.status == "completed"
    assert update_call.update.rawOutput["returncode"] == 0
    assert update_call.update.rawOutput["content"].strip() == "hello"


@pytest.mark.asyncio
async def test_tool_edit_file_overwrites(tmp_path: Path):
    target = tmp_path / "edit_me.txt"
    target.write_text("old content")

    conn = AsyncMock(spec=AgentSideConnection)
    agent = make_function_agent(conn)

    session_id = "edit-session"
    new_text = "new content with spaces"
    response = await agent.prompt(
        PromptRequest(
            sessionId=session_id,
            prompt=[text_block(f"tool:edit_file {target} {new_text}")],
        )
    )

    assert response.stopReason == "end_turn"
    assert target.read_text() == new_text
    update_call = conn.sessionUpdate.call_args_list[1][0][0]
    assert isinstance(update_call.update, ToolCallProgress)
    assert update_call.update.status == "completed"


@pytest.mark.asyncio
async def test_tool_code_search(tmp_path: Path):
    file_a = tmp_path / "a.txt"
    file_b = tmp_path / "b.txt"
    file_a.write_text("hello world\nsecond line\n")
    file_b.write_text("another file\nworld hello\n")

    conn = AsyncMock(spec=AgentSideConnection)
    agent = make_function_agent(conn)

    session_id = "search-session"
    response = await agent.prompt(
        PromptRequest(
            sessionId=session_id,
            prompt=[text_block(f"tool:code_search hello {tmp_path}")],
        )
    )

    assert response.stopReason == "end_turn"
    assert conn.sessionUpdate.call_count == 2
    update_call = conn.sessionUpdate.call_args_list[1][0][0]
    assert isinstance(update_call.update, ToolCallProgress)
    assert update_call.update.status == "completed"
    output = update_call.update.rawOutput["content"]
    assert "a.txt" in output or "b.txt" in output


@pytest.mark.asyncio
async def test_file_system_read_write(tmp_path: Path):
    conn = AsyncMock(spec=AgentSideConnection)
    agent = make_function_agent(conn)
    session = await agent.newSession(NewSessionRequest(cwd=str(tmp_path), mcpServers=[]))

    # write
    content = "hello fs"
    await agent.writeTextFile(
        WriteTextFileRequest(sessionId=session.sessionId, path="fs.txt", content=content)
    )

    # read
    response = await agent.readTextFile(
        ReadTextFileRequest(sessionId=session.sessionId, path="fs.txt", line=0, limit=10)
    )

    assert response.content == content


@pytest.mark.asyncio
async def test_plan_updates():
    conn = AsyncMock(spec=AgentSideConnection)
    agent = make_function_agent(conn)

    session_id = "plan-session"
    response = await agent.prompt(
        PromptRequest(sessionId=session_id, prompt=[text_block("plan:step one;step two")])
    )

    conn.sessionUpdate.assert_called_once()
    notification = conn.sessionUpdate.call_args[0][0]
    assert notification.sessionId == session_id
    assert hasattr(notification.update, "entries")
    assert len(notification.update.entries) == 2
    assert response.stopReason == "end_turn"


@pytest.mark.asyncio
async def test_content_blocks_use_embedded_resources():
    conn = AsyncMock(spec=AgentSideConnection)
    agent = make_function_agent(conn)

    block = resource_block(embedded_text_resource("uri:sample", "embedded text"))
    session_id = "content-session"
    response = await agent.prompt(PromptRequest(sessionId=session_id, prompt=[block]))

    assert conn.sessionUpdate.call_count >= 1
    notification = conn.sessionUpdate.call_args_list[-1][0][0]
    assert isinstance(notification.update, AgentMessageChunk)
    assert notification.update.content.text
    assert response.stopReason == "end_turn"


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


@pytest.mark.asyncio
async def test_terminal_lifecycle(tmp_path: Path):
    conn = AsyncMock(spec=AgentSideConnection)
    agent = make_function_agent(conn)
    session = await agent.newSession(NewSessionRequest(cwd=str(tmp_path), mcpServers=[]))

    # create a terminal that echoes once
    create_resp = await agent.createTerminal(
        CreateTerminalRequest(sessionId=session.sessionId, command="echo", args=["hello"])
    )
    term_id = create_resp.terminalId

    # fetch output
    out_resp = await agent.terminalOutput(
        TerminalOutputRequest(sessionId=session.sessionId, terminalId=term_id)
    )
    assert "hello" in out_resp.output

    # wait for exit
    exit_resp = await agent.waitForTerminalExit(
        WaitForTerminalExitRequest(sessionId=session.sessionId, terminalId=term_id)
    )
    assert exit_resp.exitCode == 0

    # create long-running and kill it
    create_resp2 = await agent.createTerminal(
        CreateTerminalRequest(sessionId=session.sessionId, command="sleep", args=["5"])
    )
    term_id2 = create_resp2.terminalId
    await agent.killTerminalCommand(
        KillTerminalCommandRequest(sessionId=session.sessionId, terminalId=term_id2)
    )
    exit_resp2 = await agent.waitForTerminalExit(
        WaitForTerminalExitRequest(sessionId=session.sessionId, terminalId=term_id2)
    )
    assert exit_resp2.exitCode is not None or exit_resp2.signal is not None
