"""Summarize a file without loading the full contents into context."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from isaac.agent.ai_types import ToolContext


def _resolve(base: Optional[str], target: str) -> Path:
    p = Path(target)
    if p.is_absolute():
        return p
    return Path(base or Path.cwd()) / p


async def file_summary(
    ctx: ToolContext | None = None,
    path: str = "",
    head_lines: Optional[int] = 20,
    tail_lines: Optional[int] = 20,
    cwd: Optional[str] = None,
) -> dict:
    resolved = _resolve(cwd, path)
    if not resolved.exists():
        return {"content": "", "error": f"File not found: {path}"}
    if not resolved.is_file():
        return {"content": "", "error": f"Not a file: {path}"}

    try:
        lines = resolved.read_text(encoding="utf-8", errors="ignore").splitlines()
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
        return {"content": "", "error": str(exc)}
