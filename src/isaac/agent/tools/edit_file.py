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
    file_path: str = "",
    new_content: str = "",
    create: bool = True,
    cwd: Optional[str] = None,
    **_: object,
) -> dict:
    """Overwrite a file with new content.

    Args:
        file_path: Path to the file to edit.
        new_content: Complete replacement content for the file.
        create: Whether to create the file if it does not exist.
    """
    path = _resolve(cwd, file_path)
    old_text = ""
    if path.exists():
        try:
            old_text = path.read_text(encoding="utf-8")
        except Exception:
            old_text = ""

    if not path.exists() and not create:
        return {"content": None, "error": f"File '{file_path}' does not exist", "returncode": -1}

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(new_content, encoding="utf-8")
        old_lines = old_text.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)
        diff = "".join(
            difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile=file_path,
                tofile=file_path,
                lineterm="",
            )
        )
        summary = f"Wrote {len(new_content)} bytes to {file_path}"
        content = f"{summary}\n{diff}" if diff else summary
        return {
            "content": content,
            "diff": diff,
            "error": None,
            "returncode": 0,
        }
    except Exception as exc:  # pragma: no cover - unexpected filesystem errors
        return {"content": None, "error": str(exc), "returncode": -1}
