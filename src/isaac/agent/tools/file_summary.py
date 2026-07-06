"""Summarize a file without loading the full contents into context."""

from __future__ import annotations

from typing import Optional

from isaac.agent.ai_types import ToolContext
from isaac.agent.tools.safety import (
    BinaryFileError,
    PathAccessError,
    ensure_text_file,
    resolve_workspace_path,
    sha256_file,
)


async def file_summary(
    ctx: ToolContext | None = None,
    path: str = "",
    head_lines: Optional[int] = 20,
    tail_lines: Optional[int] = 20,
    cwd: Optional[str] = None,
) -> dict:
    _ = ctx
    try:
        resolved = resolve_workspace_path(cwd, path)
    except PathAccessError as exc:
        return {"content": "", "error": str(exc)}

    if not resolved.exists():
        return {"content": "", "error": f"File not found: {path}"}
    if not resolved.is_file():
        return {"content": "", "error": f"Not a file: {path}"}

    try:
        ensure_text_file(resolved)
        lines = resolved.read_text(encoding="utf-8").splitlines()
        head = lines[: head_lines or 0] if head_lines else []
        tail = lines[-(tail_lines or 0) :] if tail_lines else []
        total = len(lines)
        parts = []
        if head:
            parts.append("Head:\n" + "\n".join(head))
        if tail:
            parts.append("Tail:\n" + "\n".join(tail))
        summary = "\n\n".join(parts)
        hash_text = sha256_file(resolved)
        hash_line = f"SHA256: {hash_text}\n" if hash_text else ""
        return {"content": f"Lines: {total}\n{hash_line}{summary}", "sha256": hash_text, "error": None}
    except BinaryFileError as exc:
        return {"content": "", "error": str(exc)}
    except Exception as exc:
        return {"content": "", "error": str(exc)}
