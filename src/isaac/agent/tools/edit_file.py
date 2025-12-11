from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any, Optional

from pydantic_ai import RunContext


def _resolve(base: Optional[str], target: str, *, allow_outside: bool) -> Path | None:
    p = Path(target)
    base_path = Path(base) if base else None
    if base_path and base_path.is_absolute():
        base_path = base_path.resolve()
    if p.is_absolute():
        resolved = p
    else:
        resolved = (base_path or Path.cwd()) / p
    resolved = resolved.resolve()
    if base_path and not allow_outside:
        try:
            resolved.relative_to(base_path)
        except ValueError:
            return None
    return resolved


async def edit_file(
    ctx: RunContext[Any] = None,
    path: str = "",
    content: str = "",
    create: bool = True,
    cwd: Optional[str] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
    allow_outside: bool = False,
    **_: object,
) -> dict:
    """Overwrite a file with new content.

    Args:
        path: Path to the file to edit.
        content: Complete replacement content for the file.
        create: Whether to create the file if it does not exist.
        start: Optional starting line (1-based) for partial replacement.
        end: Optional ending line (1-based, inclusive) for partial replacement.
    """
    if not path:
        return {"content": None, "error": "Missing required arguments: path", "returncode": -1}
    resolved = _resolve(cwd, path, allow_outside=allow_outside)
    if resolved is None:
        return {
            "content": None,
            "error": "Path is outside allowed working directory",
            "returncode": -1,
        }
    if resolved.exists() and resolved.is_dir():
        return {"content": None, "error": f"Path '{path}' is a directory", "returncode": -1}
    old_text = ""
    if resolved.exists():
        try:
            old_text = resolved.read_text(encoding="utf-8")
        except Exception:
            old_text = ""

    if not resolved.exists() and not create:
        return {"content": "", "error": f"File '{path}' does not exist", "returncode": -1}

    partial_edit = start is not None or end is not None

    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        if partial_edit:
            if not resolved.exists():
                return {
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

        resolved.write_text(new_text, encoding="utf-8")
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
            "content": result_content,
            "diff": diff,
            "new_text": new_text,
            "old_text": old_text,
            "error": None,
            "returncode": 0,
        }
    except Exception as exc:  # pragma: no cover - unexpected filesystem errors
        return {"content": "", "error": str(exc), "returncode": -1}
