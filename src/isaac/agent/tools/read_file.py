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
    file_path: str = "",
    start_line: Optional[int] = None,
    num_lines: Optional[int] = None,
    cwd: Optional[str] = None,
) -> dict:
    """Read a file with optional line range.

    Args:
        file_path (str): Path to the file to read.
        start_line (int, optional): Starting line number (1-based). If provided, num_lines must also be provided.
        num_lines (int, optional): Number of lines to read.

    Returns:
        dict: A dictionary with 'content' (str or None), 'num_tokens' (int), and 'error' (str or None).
    """
    path = _resolve(cwd, file_path)
    if not path.exists():
        return {"content": None, "num_tokens": 0, "error": f"File '{file_path}' does not exist."}

    if not path.is_file():
        return {"content": None, "num_tokens": 0, "error": f"'{file_path}' is not a file."}

    try:
        # Read full file
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Handle line range
        if start_line is not None:
            # Convert to 0-based index
            start_idx = start_line - 1
            if num_lines is None:
                selected_lines = lines[start_idx:]
            else:
                end_idx = start_idx + num_lines
                selected_lines = lines[start_idx:end_idx]
        else:
            selected_lines = lines

        content = "".join(selected_lines)

        # Estimate tokens (roughly 4 chars = 1 token)
        num_tokens = max(1, len(content) // 4)

        return {"content": content, "num_tokens": num_tokens, "error": None}
    except Exception as e:
        return {"content": None, "num_tokens": 0, "error": f"Error reading file: {str(e)}"}
