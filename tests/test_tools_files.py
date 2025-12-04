from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from acp import text_block, RequestPermissionResponse
from acp.agent.connection import AgentSideConnection
from acp.schema import ToolCallProgress, ToolCallStart, AllowedOutcome

from isaac.agent.tools.apply_patch import apply_patch
from isaac.agent.tools.list_directory import list_files
from tests.utils import make_function_agent


@pytest.mark.asyncio
async def test_tool_list_files_sends_progress(tmp_path: Path):
    (tmp_path / "file_a.txt").write_text("data")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "file_b.txt").write_text("more")

    conn = AsyncMock(spec=AgentSideConnection)
    agent = make_function_agent(conn)

    session_id = "tool-session"
    response = await agent.prompt(
        prompt=[text_block(f"tool:list_files {tmp_path}")], session_id=session_id
    )

    assert response.stop_reason == "end_turn"
    assert conn.session_update.call_count == 2

    start_call = conn.session_update.call_args_list[0].kwargs["update"]
    update_call = conn.session_update.call_args_list[1].kwargs["update"]

    assert isinstance(start_call, ToolCallStart)
    assert start_call.status == "in_progress"
    assert getattr(start_call, "session_id", session_id) == session_id

    assert isinstance(update_call, ToolCallProgress)
    assert update_call.status == "completed"
    assert update_call.raw_output["error"] is None


@pytest.mark.asyncio
async def test_tool_read_file_returns_content(tmp_path: Path):
    target = tmp_path / "sample.txt"
    target.write_text("line1\nline2\nline3\n")

    conn = AsyncMock(spec=AgentSideConnection)
    agent = make_function_agent(conn)

    session_id = "read-session"
    response = await agent.prompt(
        prompt=[text_block(f"tool:read_file {target} 2 1")], session_id=session_id
    )

    assert response.stop_reason == "end_turn"
    assert conn.session_update.call_count == 2

    update_call = conn.session_update.call_args_list[1].kwargs["update"]
    assert isinstance(update_call, ToolCallProgress)
    assert update_call.status == "completed"
    assert "line2" in update_call.raw_output["content"]


@pytest.mark.asyncio
async def test_tool_run_command_executes(tmp_path: Path):
    conn = AsyncMock(spec=AgentSideConnection)

    async def _allow(*_: object, **__: object):
        return RequestPermissionResponse(
            outcome=AllowedOutcome(option_id="allow_once", outcome="selected")
        )

    conn.request_permission = _allow  # type: ignore[attr-defined]
    agent = make_function_agent(conn)

    session_id = "cmd-session"
    response = await agent.prompt(
        prompt=[text_block("tool:run_command echo hello")], session_id=session_id
    )

    assert response.stop_reason == "end_turn"
    assert conn.session_update.call_count >= 2

    update_call = conn.session_update.call_args_list[-1].kwargs["update"]
    assert isinstance(update_call, ToolCallProgress)
    assert update_call.status == "completed"
    assert update_call.raw_output["returncode"] == 0
    assert update_call.raw_output["content"].strip() == "hello"


@pytest.mark.asyncio
async def test_tool_apply_patch(tmp_path: Path):
    target = tmp_path / "edit_me.txt"
    target.write_text("old content\n")

    patch = """--- a/edit_me.txt
+++ b/edit_me.txt
@@ -1 +1 @@
-old content
+new content
"""
    # Call tool directly to avoid DSL limitations
    result = await apply_patch(file_path=str(target), patch=patch, strip=1)
    assert result["error"] is None
    assert (result["content"] or "").strip()
    assert target.read_text() == "new content\n"


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
        prompt=[text_block(f"tool:code_search hello {tmp_path}")], session_id=session_id
    )

    assert response.stop_reason == "end_turn"
    assert conn.session_update.call_count == 2
    update_call = conn.session_update.call_args_list[1].kwargs["update"]
    assert isinstance(update_call, ToolCallProgress)
    assert update_call.status == "completed"
    output = update_call.raw_output["content"]
    assert "a.txt" in output or "b.txt" in output


def test_list_files_respects_gitignore(tmp_path: Path):
    (tmp_path / ".gitignore").write_text(".venv\n*.log\n", encoding="utf-8")
    (tmp_path / "keep.txt").write_text("ok")
    (tmp_path / "debug.log").write_text("should ignore")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "hidden.txt").write_text("hidden")

    result = asyncio.run(list_files(directory=str(tmp_path), recursive=True))

    assert result["error"] is None
    content = result.get("content") or ""
    # Should list keep.txt but not debug.log nor .venv contents
    assert "keep.txt" in content
    assert "debug.log" not in content
    assert ".venv/hidden.txt" not in content


@pytest.mark.asyncio
async def test_file_system_read_write(tmp_path: Path):
    conn = AsyncMock(spec=AgentSideConnection)
    agent = make_function_agent(conn)
    session = await agent.new_session(cwd=str(tmp_path), mcp_servers=[])

    # write
    content = "hello fs"
    await agent.write_text_file(content=content, path="fs.txt", session_id=session.session_id)

    # read
    response = await agent.read_text_file(
        path="fs.txt", session_id=session.session_id, line=0, limit=10
    )

    assert response.content == content
