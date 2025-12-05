from __future__ import annotations

from pathlib import Path

import pytest
from acp import RequestPermissionResponse, text_block
from acp.schema import AllowedOutcome
from unittest.mock import AsyncMock

from isaac.agent.brain.planning_agent import build_planning_agent
from isaac.agent.runner import register_tools
from isaac.agent.tools.apply_patch import apply_patch
from isaac.agent.tools.code_search import code_search
from isaac.agent.tools.edit_file import edit_file
from isaac.agent.tools.file_summary import file_summary
from isaac.agent.tools.list_directory import list_files
from isaac.agent.tools.read_file import read_file
from isaac.agent.tools import TOOL_HANDLERS, run_tool
from isaac.agent.tools.run_command import run_command
from tests.utils import make_function_agent
from pydantic_ai import Agent as PydanticAgent  # type: ignore
from pydantic_ai.models.test import TestModel  # type: ignore
from isaac.agent import ACPAgent


@pytest.mark.asyncio
async def test_list_files(tmp_path: Path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("b")

    result = await list_files(directory=str(tmp_path), recursive=False)
    assert result["error"] is None
    assert "a.txt" in result["content"]


@pytest.mark.asyncio
async def test_read_file_with_range(tmp_path: Path):
    target = tmp_path / "sample.txt"
    target.write_text("line1\nline2\nline3\n")

    result = await read_file(file_path=str(target), start_line=2, num_lines=1)
    assert result["error"] is None
    assert "line2" in result["content"]


@pytest.mark.asyncio
async def test_edit_file(tmp_path: Path):
    target = tmp_path / "edit.txt"
    result = await edit_file(file_path=str(target), new_content="new", create=True)
    assert result["error"] is None
    assert target.read_text() == "new"
    assert "diff" in result
    assert "+new" in result["diff"] or "new" in result["diff"]


@pytest.mark.asyncio
async def test_apply_patch(tmp_path: Path):
    target = tmp_path / "patch.txt"
    target.write_text("one\n")
    patch = """--- a/patch.txt
+++ b/patch.txt
@@ -1 +1 @@
-one
+two
"""
    result = await apply_patch(file_path=str(target), patch=patch, strip=1)
    assert result["error"] is None
    assert "two" in target.read_text()


@pytest.mark.asyncio
async def test_code_search(tmp_path: Path):
    target = tmp_path / "search.txt"
    target.write_text("hello world\n")
    result = await code_search(pattern="hello", directory=str(tmp_path))
    assert result["error"] is None
    assert "hello" in result["content"]


@pytest.mark.asyncio
async def test_run_command():
    result = await run_command("echo hi")
    assert result["error"] is None
    assert result["content"].strip() == "hi"


@pytest.mark.asyncio
async def test_file_summary(tmp_path: Path):
    target = tmp_path / "sum.txt"
    target.write_text("a\nb\nc\n")
    result = await file_summary(file_path=str(target), head_lines=1, tail_lines=1)
    assert result["error"] is None
    assert "Lines: 3" in result["content"]


@pytest.mark.asyncio
async def test_run_tool_reports_missing_args():
    result = await run_tool("edit_file")
    assert result["error"].startswith("Missing required arguments:")


@pytest.mark.asyncio
async def test_run_command_requests_permission():
    conn = AsyncMock()
    conn.request_permission = AsyncMock(
        return_value=RequestPermissionResponse(
            outcome=AllowedOutcome(option_id="reject_once", outcome="selected")
        )
    )
    agent = make_function_agent(conn)
    session_id = "perm-session"
    agent._session_modes[session_id] = "ask"
    agent._session_cwds[session_id] = Path.cwd()

    await agent._execute_run_command_with_terminal(
        session_id, tool_call_id="tc1", arguments={"command": "echo hi"}
    )

    conn.request_permission.assert_awaited()
    assert conn.session_update.await_args_list, "Expected a session_update for permission denial"
    updates = [call.kwargs["update"] for call in conn.session_update.await_args_list]  # type: ignore[attr-defined]
    failed = [u for u in updates if getattr(u, "status", "") == "failed"]
    assert failed, "Expected a failed tool update after permission denial"
    raw_out = getattr(failed[-1], "raw_output", {}) or {}
    assert raw_out.get("error") == "permission denied"


@pytest.mark.asyncio
async def test_allow_always_cached_per_command():
    conn = AsyncMock()
    conn.request_permission = AsyncMock(
        return_value=RequestPermissionResponse(
            outcome=AllowedOutcome(option_id="allow_always", outcome="selected")
        )
    )
    agent = make_function_agent(conn)
    session_id = "perm-cache"
    agent._session_modes[session_id] = "ask"
    agent._session_cwds[session_id] = Path.cwd()

    first = await agent._request_run_permission(
        session_id, tool_call_id="tc-1", command="echo cached", cwd=None
    )
    second = await agent._request_run_permission(
        session_id, tool_call_id="tc-2", command="echo cached", cwd=None
    )

    assert first is True
    assert second is True
    conn.request_permission.assert_awaited_once()


@pytest.mark.asyncio
async def test_model_tool_call_requests_permission(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Ensure ask mode prompts for permission when the model triggers run_command."""

    conn = AsyncMock()
    conn.session_update = AsyncMock()
    conn.request_permission = AsyncMock(
        return_value=RequestPermissionResponse(
            outcome=AllowedOutcome(option_id="allow_once", outcome="selected")
        )
    )

    calls: list[dict[str, object]] = []

    async def fake_run_command(command: str, cwd: str | None = None, timeout: float | None = None):
        calls.append({"command": command, "cwd": cwd, "timeout": timeout})
        return {"content": "ok", "error": None, "returncode": 0}

    monkeypatch.setattr("isaac.agent.tools.run_command", fake_run_command)
    monkeypatch.setitem(TOOL_HANDLERS, "run_command", fake_run_command)

    model = TestModel(call_tools=["run_command"], custom_output_text="done")
    ai_runner = PydanticAgent(model)
    register_tools(ai_runner)
    planning_runner = build_planning_agent(TestModel(call_tools=[]))
    agent = ACPAgent(conn, ai_runner=ai_runner, planning_runner=planning_runner)
    session = await agent.new_session(cwd=str(tmp_path), mcp_servers=[])

    response = await agent.prompt(
        prompt=[text_block("run a command")], session_id=session.session_id
    )

    conn.request_permission.assert_awaited()
    assert calls, "Expected run_command to be invoked by the model"
    assert response.stop_reason == "end_turn"
