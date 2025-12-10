from typing import Any, Optional

from pathlib import Path
from pydantic_ai import RunContext


def _resolve(base: Optional[str], target: str) -> Path:
    p = Path(target)
    if p.is_absolute():
        return p
    return Path(base or Path.cwd()) / p


async def read_file(
    ctx: RunContext[Any] = None,
    path: str = "",
    start: Optional[int] = None,
    lines: Optional[int] = None,
    cwd: Optional[str] = None,
) -> dict:
    """Read a file with optional line range.

    Args:
        path (str): Path to the file to read.
        start (int, optional): Starting line number (1-based). If provided, lines must also be provided.
        lines (int, optional): Number of lines to read.

    Returns:
        dict: A dictionary with 'content' (str or None), 'num_tokens' (int), and 'error' (str or None).
    """
    resolved = _resolve(cwd, path)
    if not resolved.exists():
        return {"content": "", "num_tokens": 0, "error": f"File '{path}' does not exist."}

    if not resolved.is_file():
        return {"content": "", "num_tokens": 0, "error": f"'{path}' is not a file."}

    try:
        # Read full file
        with open(resolved, "r", encoding="utf-8") as f:
            file_lines = f.readlines()

        # Handle line range
        if start is not None:
            # Convert to 0-based index
            start_idx = start - 1
            if lines is None:
                selected_lines = file_lines[start_idx:]
            else:
                end_idx = start_idx + lines
                selected_lines = file_lines[start_idx:end_idx]
        else:
            selected_lines = file_lines

        content = "".join(selected_lines)

        # Estimate tokens (roughly 4 chars = 1 token)
        num_tokens = max(1, len(content) // 4)

        return {"content": content, "num_tokens": num_tokens, "error": None}
    except Exception as e:
        return {"content": "", "num_tokens": 0, "error": f"Error reading file: {str(e)}"}
