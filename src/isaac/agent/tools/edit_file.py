from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any, Optional

from pydantic_ai import RunContext


def _resolve(base: Optional[str], target: str) -> Path:
    p = Path(target)
    if p.is_absolute():
        return p
    return Path(base or Path.cwd()) / p


async def edit_file(
    ctx: RunContext[Any] = None,
    path: str = "",
    content: str = "",
    create: bool = True,
    cwd: Optional[str] = None,
    **_: object,
) -> dict:
    """Overwrite a file with new content.

    Args:
        path: Path to the file to edit.
        content: Complete replacement content for the file.
        create: Whether to create the file if it does not exist.
    """
    if not path:
        return {"content": None, "error": "Missing required arguments: path", "returncode": -1}
    resolved = _resolve(cwd, path)
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

    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        old_lines = old_text.splitlines(keepends=True)
        new_lines = content.splitlines(keepends=True)
        diff = "".join(
            difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile=path,
                tofile=path,
                lineterm="",
            )
        )
        summary = f"Wrote {len(content)} bytes to {path}"
        result_content = f"{summary}\n{diff}" if diff else summary
        return {
            "content": result_content,
            "diff": diff,
            "error": None,
            "returncode": 0,
        }
    except Exception as exc:  # pragma: no cover - unexpected filesystem errors
        return {"content": "", "error": str(exc), "returncode": -1}
