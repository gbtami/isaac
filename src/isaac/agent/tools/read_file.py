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


async def read_file(
    ctx: ToolContext | None = None,
    path: str = "",
    start: Optional[int] = None,
    lines: Optional[int] = None,
    cwd: Optional[str] = None,
) -> dict:
    """Read a text file with optional line range."""

    _ = ctx
    try:
        resolved = resolve_workspace_path(cwd, path)
    except PathAccessError as exc:
        return {"content": "", "num_tokens": 0, "error": str(exc)}

    if not resolved.exists():
        return {"content": "", "num_tokens": 0, "error": f"File '{path}' does not exist."}

    if not resolved.is_file():
        return {"content": "", "num_tokens": 0, "error": f"'{path}' is not a file."}

    try:
        ensure_text_file(resolved)
        file_lines = resolved.read_text(encoding="utf-8").splitlines(keepends=True)

        if start is not None:
            start_idx = max(start - 1, 0)
            if lines is None:
                selected_lines = file_lines[start_idx:]
            else:
                selected_lines = file_lines[start_idx : start_idx + max(lines, 0)]
        else:
            selected_lines = file_lines

        content = "".join(selected_lines)
        num_tokens = max(1, len(content) // 4)

        return {
            "content": content,
            "num_tokens": num_tokens,
            "sha256": sha256_file(resolved),
            "error": None,
        }
    except BinaryFileError as exc:
        return {"content": "", "num_tokens": 0, "error": str(exc)}
    except Exception as e:
        return {"content": "", "num_tokens": 0, "error": f"Error reading file: {str(e)}"}
