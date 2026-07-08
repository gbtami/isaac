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
from pydantic_ai import DeferredToolRequests  # type: ignore
from pydantic_ai.models.test import TestModel  # type: ignore

from isaac.agent.tools import build_isaac_tools_capability
from isaac.agent.tools.apply_patch import apply_patch
from isaac.agent.tools.edit_file import edit_file
from isaac.agent.tools.list_files import list_files
from isaac.agent.tools.read_file import read_file
from isaac.agent.tools.run_command import run_command
from isaac.agent.tools.safety import sha256_file
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
        ),
        output_type=[str, DeferredToolRequests],
        toolsets=(),
        capabilities=[build_isaac_tools_capability()],
    )
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
async def test_direct_acp_tool_call_uses_session_cwd(tmp_path: Path):
    target = tmp_path / "relative.txt"
    target.write_text("from session cwd")

    conn = AsyncMock(spec=AgentSideConnection)
    conn.session_update = AsyncMock()
    agent = make_function_agent(conn)
    session = await agent.new_session(cwd=str(tmp_path), mcp_servers=[])

    await agent._execute_tool(
        session_id=session.session_id,
        tool_name="read_file",
        tool_call_id="tc-relative-read",
        arguments={"path": "relative.txt"},
    )

    progress = [
        call.kwargs["update"]
        for call in conn.session_update.call_args_list
        if isinstance(call.kwargs.get("update"), ToolCallProgress)
    ]
    assert progress
    assert progress[-1].status == "completed"
    assert progress[-1].raw_output["error"] is None
    assert progress[-1].raw_output["content"] == "from session cwd"


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
        ),
        output_type=[str, DeferredToolRequests],
        toolsets=(),
        capabilities=[build_isaac_tools_capability()],
    )
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
        ),
        output_type=[str, DeferredToolRequests],
        toolsets=(),
        capabilities=[build_isaac_tools_capability()],
    )
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
async def test_model_tool_read_file_uses_session_cwd_for_relative_path(tmp_path: Path):
    target = tmp_path / "relative_model.txt"
    target.write_text("from model session cwd", encoding="utf-8")

    conn = AsyncMock(spec=AgentSideConnection)
    conn.session_update = AsyncMock()
    agent = make_function_agent(conn)
    session = await agent.new_session(cwd=str(tmp_path), mcp_servers=[])

    runner = PydanticAgent(
        FixedArgsModel(
            fixed_args={"read_file": {"path": "relative_model.txt"}},
            call_tools=["read_file"],
            custom_output_text="done",
        ),
        output_type=[str, DeferredToolRequests],
        toolsets=(),
        capabilities=[build_isaac_tools_capability()],
    )
    agent._prompt_handler.set_session_runner(session.session_id, runner)  # type: ignore[attr-defined]

    response = await agent.prompt(prompt=[text_block("read relative file")], session_id=session.session_id)

    assert response.stop_reason == "end_turn"
    progress = [
        call.kwargs["update"]
        for call in conn.session_update.call_args_list
        if isinstance(call.kwargs.get("update"), ToolCallProgress)
    ]
    assert progress
    assert progress[-1].status == "completed"
    assert progress[-1].raw_output["error"] is None
    assert progress[-1].raw_output["content"] == "from model session cwd"


@pytest.mark.asyncio
async def test_direct_tool_can_read_absolute_path_from_additional_directory(tmp_path: Path):
    workspace = tmp_path / "workspace"
    docs = tmp_path / "docs"
    workspace.mkdir()
    docs.mkdir()
    target = docs / "notes.txt"
    target.write_text("shared context", encoding="utf-8")

    conn = AsyncMock(spec=AgentSideConnection)
    conn.session_update = AsyncMock()
    agent = make_function_agent(conn)
    session = await agent.new_session(
        cwd=str(workspace),
        additional_directories=[str(docs)],
        mcp_servers=[],
    )

    await agent._execute_tool(
        session_id=session.session_id,
        tool_name="read_file",
        tool_call_id="tc-additional-read",
        arguments={"path": str(target)},
    )

    progress = [
        call.kwargs["update"]
        for call in conn.session_update.call_args_list
        if isinstance(call.kwargs.get("update"), ToolCallProgress)
    ]
    assert progress
    assert progress[-1].status == "completed"
    assert progress[-1].raw_output["error"] is None
    assert progress[-1].raw_output["content"] == "shared context"


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
        ),
        output_type=[str, DeferredToolRequests],
        toolsets=(),
        capabilities=[build_isaac_tools_capability()],
    )
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


@pytest.mark.asyncio
async def test_read_file_blocks_symlink_escape(tmp_path: Path):
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    (workspace / "link").symlink_to(outside, target_is_directory=True)

    result = await read_file(path="link/secret.txt", cwd=str(workspace))

    assert result["error"] == "Path is outside allowed working directory"
    assert result["content"] == ""


@pytest.mark.asyncio
async def test_list_files_hides_outside_symlink(tmp_path: Path):
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    (workspace / "safe.txt").write_text("ok", encoding="utf-8")
    (workspace / "outside-link").symlink_to(outside, target_is_directory=True)

    result = await list_files(directory=".", recursive=False, cwd=str(workspace))

    assert result["error"] is None
    assert "safe.txt" in result["content"]
    assert "outside-link" not in result["content"]


@pytest.mark.asyncio
async def test_edit_file_refuses_protected_paths(tmp_path: Path):
    result = await edit_file(path=".env", content="TOKEN=secret\n", cwd=str(tmp_path))

    assert result["error"]
    assert "protected path" in result["error"]
    assert not (tmp_path / ".env").exists()


@pytest.mark.asyncio
async def test_text_tools_refuse_binary_files(tmp_path: Path):
    target = tmp_path / "image.bin"
    target.write_bytes(b"abc\x00def")

    read_result = await read_file(path="image.bin", cwd=str(tmp_path))
    edit_result = await edit_file(path="image.bin", content="text", cwd=str(tmp_path))

    assert "binary file" in read_result["error"]
    assert "binary file" in edit_result["error"]
    assert target.read_bytes() == b"abc\x00def"


@pytest.mark.asyncio
async def test_edit_file_expected_sha256_blocks_stale_write(tmp_path: Path):
    target = tmp_path / "guarded.txt"
    target.write_text("old\n", encoding="utf-8")
    stale_hash = sha256_file(target)
    target.write_text("changed\n", encoding="utf-8")

    result = await edit_file(
        path="guarded.txt",
        content="new\n",
        cwd=str(tmp_path),
        expected_sha256=stale_hash,
    )

    assert result["error"] == "File changed since it was read; expected_sha256 does not match"
    assert target.read_text(encoding="utf-8") == "changed\n"


@pytest.mark.asyncio
async def test_run_command_caps_stdout_with_metadata(tmp_path: Path):
    result = await run_command(
        command="python -c 'print(\"x\" * 80)'",
        cwd=str(tmp_path),
        max_output_chars=20,
    )

    assert result["error"] is None
    assert result["returncode"] == 0
    assert result["stdout_truncated"] is True
    assert result["stderr_truncated"] is False
    assert result["truncated"] is True
    assert result["max_output_chars"] == 20
    assert "[truncated after 20 characters]" in result["content"]


@pytest.mark.asyncio
async def test_run_command_keeps_successful_stderr_as_diagnostic(tmp_path: Path):
    result = await run_command(
        command="python -c 'import sys; print(\"warn\", file=sys.stderr)'",
        cwd=str(tmp_path),
        max_output_chars=100,
    )

    assert result["returncode"] == 0
    assert result["error"] is None
    assert result["stderr"] == "warn"


@pytest.mark.asyncio
async def test_run_command_failed_stderr_becomes_error(tmp_path: Path):
    result = await run_command(
        command="python -c 'import sys; print(\"boom\", file=sys.stderr); sys.exit(3)'",
        cwd=str(tmp_path),
        max_output_chars=100,
    )

    assert result["returncode"] == 3
    assert result["stderr"] == "boom"
    assert result["error"] == "boom"

@pytest.mark.asyncio
async def test_run_command_shell_policy_env_allowlist(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("ISAAC_SHELL_ALLOWLIST", r"^echo\b")

    allowed = await run_command(command="echo ok", cwd=str(tmp_path))
    blocked = await run_command(command="python -V", cwd=str(tmp_path))

    assert allowed["error"] is None
    assert allowed["content"] == "ok"
    assert blocked["error"] == "Command does not match ISAAC_SHELL_ALLOWLIST"


@pytest.mark.asyncio
async def test_run_command_blocks_catastrophic_patterns():
    result = await run_command(command="rm -rf /")

    assert result["returncode"] == -1
    assert "catastrophic deny pattern" in result["error"]


@pytest.mark.asyncio
async def test_edit_file_rejects_symlink_write_target(tmp_path: Path):
    real_target = tmp_path / "real.txt"
    real_target.write_text("original", encoding="utf-8")
    link = tmp_path / "link.txt"
    link.symlink_to(real_target)

    result = await edit_file(path="link.txt", content="changed", cwd=str(tmp_path))

    assert result["error"]
    assert "symlink" in result["error"].lower()
    assert real_target.read_text(encoding="utf-8") == "original"


@pytest.mark.asyncio
async def test_apply_patch_rejects_headers_for_different_file(tmp_path: Path):
    target = tmp_path / "target.txt"
    target.write_text("one\n", encoding="utf-8")
    other = tmp_path / "other.txt"
    other.write_text("one\n", encoding="utf-8")

    patch = """--- other.txt
+++ other.txt
@@ -1 +1 @@
-one
+two
"""
    result = await apply_patch(path=str(target), patch=patch, strip=0)

    assert result["error"]
    assert "may only modify target.txt" in result["error"]
    assert target.read_text(encoding="utf-8") == "one\n"
    assert other.read_text(encoding="utf-8") == "one\n"


@pytest.mark.asyncio
async def test_apply_patch_rejects_parent_escape_header(tmp_path: Path):
    target = tmp_path / "target.txt"
    target.write_text("one\n", encoding="utf-8")

    patch = """--- ../target.txt
+++ ../target.txt
@@ -1 +1 @@
-one
+two
"""
    result = await apply_patch(path=str(target), patch=patch, strip=0)

    assert result["error"]
    assert "escapes" in result["error"]
    assert target.read_text(encoding="utf-8") == "one\n"
