from __future__ import annotations

from pathlib import Path
from typing import Optional

from isaac.agent.ai_types import ToolContext
from isaac.agent.tools.safety import (
    BinaryFileError,
    PathAccessError,
    ensure_text_file,
    resolve_workspace_path,
    sha256_file,
)

DEFAULT_READ_MAX_LINES = 400
MAX_READ_MAX_LINES = 1_000


def _bounded_max_lines(value: int | None) -> int:
    if value is None:
        return DEFAULT_READ_MAX_LINES
    return max(1, min(int(value), MAX_READ_MAX_LINES))


async def read_file(
    ctx: ToolContext | None = None,
    path: str = "",
    start: Optional[int] = None,
    lines: Optional[int] = None,
    max_lines: Optional[int] = None,
    cwd: Optional[str] = None,
    session_cwd: str | Path | None = None,
    additional_directories: tuple[str | Path, ...] = (),
) -> dict:
    """Read a text file with optional line range and bounded output.

    The default caps full-file reads to keep model-facing tool results useful in
    long coding sessions. Use ``start`` plus ``lines`` or the returned
    ``next_start`` value to page through large files deliberately.
    """

    _ = ctx
    try:
        resolved = resolve_workspace_path(session_cwd or cwd, path, additional_directories=additional_directories)
    except PathAccessError as exc:
        return {"content": "", "num_tokens": 0, "error": str(exc)}

    if not resolved.exists():
        return {"content": "", "num_tokens": 0, "error": f"File '{path}' does not exist."}

    if not resolved.is_file():
        return {"content": "", "num_tokens": 0, "error": f"'{path}' is not a file."}

    try:
        ensure_text_file(resolved)
        file_lines = resolved.read_text(encoding="utf-8").splitlines(keepends=True)
        total_lines = len(file_lines)

        start_idx = max((start or 1) - 1, 0)
        if start_idx >= total_lines:
            selected_lines: list[str] = []
            truncated = False
            next_start = None
        else:
            if lines is not None and max_lines is not None:
                requested_count = min(lines, max_lines)
            else:
                requested_count = lines if lines is not None else max_lines
            cap = _bounded_max_lines(requested_count)
            available = file_lines[start_idx:]
            selected_lines = available[:cap]
            truncated = len(available) > len(selected_lines)
            next_start = start_idx + len(selected_lines) + 1 if truncated else None

        content = "".join(selected_lines)
        num_tokens = max(1, len(content) // 4) if content else 0
        end_line = start_idx + len(selected_lines) if selected_lines else start_idx

        response = {
            "content": content,
            "num_tokens": num_tokens,
            "sha256": sha256_file(resolved),
            "error": None,
            "path": path,
            "total_lines": total_lines,
            "start_line": start_idx + 1 if total_lines else 0,
            "end_line": end_line,
            "lines_returned": len(selected_lines),
            "truncated": truncated,
        }
        if next_start is not None:
            response["next_start"] = next_start
            response["hint"] = f"Output truncated. Continue with start={next_start}."
        return response
    except BinaryFileError as exc:
        return {"content": "", "num_tokens": 0, "error": str(exc)}
    except Exception as e:
        return {"content": "", "num_tokens": 0, "error": f"Error reading file: {str(e)}"}
