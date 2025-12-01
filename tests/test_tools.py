from __future__ import annotations

from pathlib import Path

import pytest

from isaac.agent.tools.apply_patch import apply_patch
from isaac.agent.tools.code_search import code_search
from isaac.agent.tools.edit_file import edit_file
from isaac.agent.tools.file_summary import file_summary
from isaac.agent.tools.list_directory import list_files
from isaac.agent.tools.read_file import read_file
from isaac.agent.tools.run_command import run_command


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
