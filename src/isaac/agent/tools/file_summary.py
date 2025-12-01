"""Summarize a file without loading the full contents into context."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


async def file_summary(
    file_path: str, head_lines: Optional[int] = 20, tail_lines: Optional[int] = 20
) -> dict:
    path = Path(file_path)
    if not path.exists():
        return {"content": None, "error": f"File not found: {file_path}"}
    if not path.is_file():
        return {"content": None, "error": f"Not a file: {file_path}"}

    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        head = lines[: head_lines or 0] if head_lines else []
        tail = lines[-(tail_lines or 0) :] if tail_lines else []
        total = len(lines)
        parts = []
        if head:
            parts.append("Head:\n" + "\n".join(head))
        if tail:
            parts.append("Tail:\n" + "\n".join(tail))
        summary = "\n\n".join(parts)
        return {"content": f"Lines: {total}\n{summary}", "error": None}
    except Exception as exc:
        return {"content": None, "error": str(exc)}
