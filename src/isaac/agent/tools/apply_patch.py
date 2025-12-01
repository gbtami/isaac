"""Apply a unified diff patch to a file."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Optional


async def apply_patch(file_path: str, patch: str, strip: Optional[int] = None) -> dict:
    """Apply a unified diff patch to a file using the `patch` command."""
    path = Path(file_path)
    if not path.exists():
        return {"content": None, "error": f"File not found: {file_path}"}

    strip_val = str(strip if strip is not None else 0)
    try:
        with tempfile.NamedTemporaryFile("w+", delete=False) as tmp:
            tmp.write(patch)
            tmp.flush()
            tmp_path = tmp.name

        proc = await asyncio.create_subprocess_exec(
            "patch",
            f"-p{strip_val}",
            "-i",
            tmp_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(path.parent),
        )
        stdout, stderr = await proc.communicate()
    except FileNotFoundError:
        return {"content": None, "error": "`patch` command not found"}
    except Exception as exc:  # pragma: no cover - defensive
        return {"content": None, "error": str(exc)}

    stdout_text = stdout.decode(errors="ignore") if stdout else ""
    stderr_text = stderr.decode(errors="ignore") if stderr else ""
    if proc.returncode != 0:
        return {"content": stdout_text, "error": stderr_text or f"Patch failed ({proc.returncode})"}
    return {"content": stdout_text or "Patch applied", "error": None}
