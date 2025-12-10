"""Apply a unified diff patch to a file."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any, Optional

from pydantic_ai import RunContext


def _resolve(base: Optional[str], target: str) -> Path:
    p = Path(target)
    if p.is_absolute():
        return p
    return Path(base or Path.cwd()) / p


async def apply_patch(
    ctx: RunContext[Any] = None,
    path: str = "",
    patch: str = "",
    strip: Optional[int] = None,
    cwd: Optional[str] = None,
) -> dict:
    """Apply a unified diff patch to a file using the `patch` command."""
    resolved = _resolve(cwd, path)
    if not resolved.exists():
        return {"content": "", "error": f"File not found: {path}"}

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
            cwd=str(resolved.parent),
        )
        stdout, stderr = await proc.communicate()
    except FileNotFoundError:
        return {"content": "", "error": "`patch` command not found"}
    except Exception as exc:  # pragma: no cover - defensive
        return {"content": "", "error": str(exc)}

    stdout_text = stdout.decode(errors="ignore") if stdout else ""
    stderr_text = stderr.decode(errors="ignore") if stderr else ""
    if proc.returncode != 0:
        return {"content": stdout_text, "error": stderr_text or f"Patch failed ({proc.returncode})"}
    return {"content": stdout_text or "Patch applied", "error": None}
