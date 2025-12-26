from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock
from typing import Any

import pytest
from acp import RequestPermissionResponse, text_block
from acp.agent.connection import AgentSideConnection
from acp.schema import AllowedOutcome, ToolCallProgress, ToolCallStart
from pydantic_ai import Agent as PydanticAgent  # type: ignore
from pydantic_ai.models.test import TestModel  # type: ignore

from isaac.agent.tools import register_tools
from isaac.agent.tools.apply_patch import apply_patch
from isaac.agent.tools.edit_file import edit_file
from isaac.agent.tools.list_files import list_files
from tests.utils import make_function_agent


class FixedArgsModel(TestModel):  # type: ignore[misc]
    """TestModel that returns predetermined tool arguments per tool name."""

    def __init__(self, fixed_args: dict[str, dict[str, object]], **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._fixed_args = fixed_args

    def gen_tool_args(self, tool_def: object) -> dict:
        name = getattr(tool_def, "name", "")
        if name in self._fixed_args:
            return self._fixed_args[name]
        return super().gen_tool_args(tool_def)


@pytest.mark.asyncio
async def test_tool_list_files_sends_progress(tmp_path: Path):
    (tmp_path / "file_a.txt").write_text("data")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "file_b.txt").write_text("more")

    conn = AsyncMock(spec=AgentSideConnection)
    conn.session_update = AsyncMock()
    agent = make_function_agent(conn)
    session = await agent.new_session(cwd=str(tmp_path), mcp_servers=[])

    runner = PydanticAgent(
        FixedArgsModel(
            fixed_args={"list_files": {"directory": str(tmp_path), "recursive": True}},
            call_tools=["list_files"],
            custom_output_text="done",
        )
    )
    register_tools(runner)
    agent._prompt_handler.set_session_runner(session.session_id, runner)  # type: ignore[attr-defined]

    response = await agent.prompt(prompt=[text_block("list files")], session_id=session.session_id)

    assert response.stop_reason == "end_turn"
    tool_updates = [
        call.kwargs["update"]
        for call in conn.session_update.call_args_list
        if isinstance(call.kwargs.get("update"), (ToolCallStart, ToolCallProgress))
    ]
    start_call = next(u for u in tool_updates if isinstance(u, ToolCallStart))
    update_call = next(u for u in tool_updates if isinstance(u, ToolCallProgress))

    assert start_call.status == "in_progress"
    assert update_call.status == "completed"
    assert update_call.raw_output["error"] is None


@pytest.mark.asyncio
async def test_tool_read_file_returns_content(tmp_path: Path):
    target = tmp_path / "sample.txt"
    target.write_text("line1\nline2\nline3\n")

    conn = AsyncMock(spec=AgentSideConnection)
    conn.session_update = AsyncMock()
    agent = make_function_agent(conn)
    session = await agent.new_session(cwd=str(tmp_path), mcp_servers=[])

    runner = PydanticAgent(
        FixedArgsModel(
            fixed_args={
                "read_file": {
                    "path": str(target),
                    "start": 2,
                    "lines": 1,
                }
            },
            call_tools=["read_file"],
            custom_output_text="done",
        )
    )
    register_tools(runner)
    agent._prompt_handler.set_session_runner(session.session_id, runner)  # type: ignore[attr-defined]

    response = await agent.prompt(prompt=[text_block("read file")], session_id=session.session_id)

    assert response.stop_reason == "end_turn"
    progress = [
        call.kwargs["update"]
        for call in conn.session_update.call_args_list
        if isinstance(call.kwargs.get("update"), ToolCallProgress)
    ]
    assert progress, "Expected tool progress"
    assert progress[-1].status == "completed"
    assert "line2" in progress[-1].raw_output["content"]


@pytest.mark.asyncio
async def test_tool_run_command_executes(tmp_path: Path):
    conn = AsyncMock(spec=AgentSideConnection)
    conn.session_update = AsyncMock()
    conn.request_permission = AsyncMock(
        return_value=RequestPermissionResponse(outcome=AllowedOutcome(option_id="allow_once", outcome="selected"))
    )
    agent = make_function_agent(conn)
    session = await agent.new_session(cwd=str(tmp_path), mcp_servers=[])

    runner = PydanticAgent(
        FixedArgsModel(
            fixed_args={"run_command": {"command": "echo hello", "cwd": None}},
            call_tools=["run_command"],
            custom_output_text="done",
        )
    )
    register_tools(runner)
    agent._prompt_handler.set_session_runner(session.session_id, runner)  # type: ignore[attr-defined]

    response = await agent.prompt(prompt=[text_block("run command")], session_id=session.session_id)

    assert response.stop_reason == "end_turn"
    progress = [
        call.kwargs["update"]
        for call in conn.session_update.call_args_list
        if isinstance(call.kwargs.get("update"), ToolCallProgress)
    ]
    assert progress, "Expected tool progress"
    final = progress[-1]
    assert final.status == "completed"
    assert final.raw_output["returncode"] == 0
    assert final.raw_output["content"].strip() == "hello"


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
    result = await apply_patch(path=str(target), patch=patch, strip=1)
    assert result["error"] is None
    assert (result["content"] or "").strip()
    assert target.read_text() == "new content\n"


@pytest.mark.asyncio
async def test_tool_apply_patch_dedents_indented_patch(tmp_path: Path):
    target = tmp_path / "demo.txt"
    target.write_text("line one\nline two\n")

    patch = """    --- demo.txt
    +++ demo.txt
    @@ -1,2 +1,4 @@
    -line one
    -line two
    +line one
    +line 1.5
    +line two
    +line three
    """
    result = await apply_patch(path=str(target), patch=patch, strip=0)
    assert result["error"] is None
    assert "line 1.5" in target.read_text()


@pytest.mark.asyncio
async def test_edit_file_partial_replace(tmp_path: Path):
    target = tmp_path / "partial.txt"
    target.write_text("line1\nline2\nline3\n")

    result = await edit_file(
        path=str(target),
        content="LINE2\nLINE3\n",
        start=2,
        end=3,
        cwd=str(tmp_path),
    )

    assert result["error"] is None
    assert target.read_text() == "line1\nLINE2\nLINE3\n"
    assert "LINE2" in (result.get("diff") or "")


@pytest.mark.asyncio
async def test_edit_file_blocks_outside_cwd(tmp_path: Path):
    target = tmp_path / "outside.txt"
    result = await edit_file(path="../outside.txt", content="data", cwd=str(tmp_path))
    assert result["error"]
    assert not target.exists()


@pytest.mark.asyncio
async def test_tool_code_search(tmp_path: Path):
    file_a = tmp_path / "a.txt"
    file_b = tmp_path / "b.txt"
    file_a.write_text("hello world\nsecond line\n")
    file_b.write_text("another file\nworld hello\n")

    conn = AsyncMock(spec=AgentSideConnection)
    conn.session_update = AsyncMock()
    agent = make_function_agent(conn)
    session = await agent.new_session(cwd=str(tmp_path), mcp_servers=[])

    runner = PydanticAgent(
        FixedArgsModel(
            fixed_args={"code_search": {"pattern": "hello", "directory": str(tmp_path)}},
            call_tools=["code_search"],
            custom_output_text="done",
        )
    )
    register_tools(runner)
    agent._prompt_handler.set_session_runner(session.session_id, runner)  # type: ignore[attr-defined]

    response = await agent.prompt(prompt=[text_block("search hello")], session_id=session.session_id)

    assert response.stop_reason == "end_turn"
    progress = [
        call.kwargs["update"]
        for call in conn.session_update.call_args_list
        if isinstance(call.kwargs.get("update"), ToolCallProgress)
    ]
    assert progress, "Expected tool progress"
    update_call = progress[-1]
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


def test_list_files_truncates_and_skips_default_ignores(tmp_path: Path):
    for idx in range(510):
        (tmp_path / f"file_{idx}.txt").write_text("x", encoding="utf-8")
    ignored = tmp_path / ".uv-cache"
    ignored.mkdir()
    (ignored / "junk.txt").write_text("junk", encoding="utf-8")

    result = asyncio.run(list_files(directory=str(tmp_path), recursive=True))

    assert result["error"] is None
    content = result.get("content") or ""
    lines = [ln for ln in content.splitlines() if ln.strip()]
    assert len(lines) <= 501  # 500 entries + optional truncation marker
    assert "[truncated]" in content
    assert result.get("truncated") is True
    assert ".uv-cache" not in content


@pytest.mark.asyncio
async def test_tool_output_is_truncated_before_sending(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    conn = AsyncMock(spec=AgentSideConnection)
    conn.session_update = AsyncMock()
    agent = make_function_agent(conn)
    session = await agent.new_session(cwd=str(tmp_path), mcp_servers=[])

    long_content = "x" * (agent._tool_output_limit + 2000)

    async def fake_run_tool(tool_name: str, **kwargs: Any) -> dict[str, Any]:
        return {"content": long_content, "error": None}

    monkeypatch.setattr("isaac.agent.tool_execution.run_tool", fake_run_tool)

    await agent._execute_tool(
        session_id=session.session_id,
        tool_name="list_files",
        tool_call_id="tc-big",
        arguments={},
    )

    updates = [call.kwargs["update"] for call in conn.session_update.call_args_list]
    completed = [u for u in updates if getattr(u, "status", "") == "completed"]
    assert completed, "Expected completed tool update"
    raw_out = getattr(completed[-1], "raw_output", {}) or {}
    assert raw_out.get("truncated") is True
    content = raw_out.get("content") or ""
    assert len(content) <= agent._tool_output_limit
    assert content.endswith("[truncated]")


@pytest.mark.asyncio
async def test_file_system_read_write(tmp_path: Path):
    conn = AsyncMock(spec=AgentSideConnection)
    agent = make_function_agent(conn)
    session = await agent.new_session(cwd=str(tmp_path), mcp_servers=[])

    # write
    content = "hello fs"
    await agent.write_text_file(content=content, path="fs.txt", session_id=session.session_id)

    # read
    response = await agent.read_text_file(path="fs.txt", session_id=session.session_id, line=0, limit=10)

    assert response.content == content
