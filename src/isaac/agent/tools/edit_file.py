from __future__ import annotations

import difflib
import os
import tempfile
from pathlib import Path
from typing import Optional

from isaac.agent.ai_types import ToolContext
from isaac.agent.tools.safety import (
    BinaryFileError,
    PathAccessError,
    ProtectedPathError,
    ensure_no_symlink_in_write_path,
    ensure_text_target,
    ensure_text_write_size,
    resolve_workspace_path,
    sha256_file,
)


def _atomic_write_text(path: Path, content: str) -> None:
    tmp_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            delete=False,
            dir=str(path.parent),
            encoding="utf-8",
        ) as tmp:
            tmp_name = tmp.name
            tmp.write(content)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_name, path)
    finally:
        if tmp_name:
            Path(tmp_name).unlink(missing_ok=True)


async def edit_file(
    ctx: ToolContext | None = None,
    path: str = "",
    content: str = "",
    create: bool = True,
    cwd: Optional[str] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
    allow_outside: bool = False,
    expected_sha256: Optional[str] = None,
    session_cwd: str | Path | None = None,
    additional_directories: tuple[str | Path, ...] = (),
    **_: object,
) -> dict:
    """Overwrite a text file with new content."""

    _ = ctx
    if not path:
        return {
            "path": path,
            "content": None,
            "error": "Missing required arguments: path",
            "returncode": -1,
        }
    try:
        base = session_cwd or cwd
        ensure_no_symlink_in_write_path(base, path, additional_directories=additional_directories)
        resolved = resolve_workspace_path(
            base,
            path,
            allow_outside=allow_outside,
            additional_directories=additional_directories,
        )
        ensure_text_target(resolved, base)
        ensure_text_write_size(content)
    except (PathAccessError, ProtectedPathError, BinaryFileError) as exc:
        return {
            "path": path,
            "content": None,
            "error": str(exc),
            "returncode": -1,
        }
    if resolved.exists() and resolved.is_dir():
        return {
            "path": path,
            "content": None,
            "error": f"Path '{path}' is a directory",
            "returncode": -1,
        }
    old_text = ""
    old_hash = sha256_file(resolved) if resolved.exists() else None
    if resolved.exists():
        try:
            old_text = resolved.read_text(encoding="utf-8")
        except Exception:
            old_text = ""

    if expected_sha256 is not None and old_hash != expected_sha256:
        return {
            "path": path,
            "content": "",
            "error": "File changed since it was read; expected_sha256 does not match",
            "returncode": -1,
            "sha256": old_hash,
        }

    if not resolved.exists() and not create:
        return {
            "path": path,
            "content": "",
            "error": f"File '{path}' does not exist",
            "returncode": -1,
        }

    partial_edit = start is not None or end is not None

    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        if partial_edit:
            if not resolved.exists():
                return {
                    "path": path,
                    "content": "",
                    "error": f"File '{path}' does not exist for partial edit",
                    "returncode": -1,
                }
            old_lines = old_text.splitlines(keepends=True)
            new_lines = content.splitlines(keepends=True)
            start_idx = max((start or 1) - 1, 0)
            end_idx = (end - 1) if end is not None else start_idx + len(new_lines) - 1
            end_idx = max(end_idx, start_idx - 1)
            if start_idx > len(old_lines):
                old_lines.extend(["\n"] * (start_idx - len(old_lines)))
            replacement = new_lines if new_lines else []
            new_contents = old_lines[:start_idx] + replacement + old_lines[end_idx + 1 :]
            new_text = "".join(new_contents)
        else:
            new_text = content

        ensure_text_write_size(new_text)
        _atomic_write_text(resolved, new_text)
        new_hash = sha256_file(resolved)
        old_lines = old_text.splitlines(keepends=True)
        new_lines = new_text.splitlines(keepends=True)
        diff = "".join(
            difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile=path,
                tofile=path,
                lineterm="",
            )
        )
        summary = (
            f"Replaced lines {start or '?'}-{end or '?'} in {path}"
            if partial_edit
            else f"Wrote {len(new_text)} bytes to {path}"
        )
        result_content = f"{summary}\n{diff}" if diff else summary
        return {
            "path": path,
            "content": result_content,
            "diff": diff,
            "new_text": new_text,
            "old_text": old_text,
            "old_sha256": old_hash,
            "sha256": new_hash,
            "error": None,
            "returncode": 0,
        }
    except Exception as exc:  # pragma: no cover - unexpected filesystem errors
        return {"path": path, "content": "", "error": str(exc), "returncode": -1}
