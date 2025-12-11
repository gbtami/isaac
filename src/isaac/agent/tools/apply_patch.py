"""Apply a unified diff patch to a file."""

from __future__ import annotations

import asyncio
import tempfile
import textwrap
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
    **_: object,
) -> dict:
    """Apply a unified diff patch to a file using the `patch` command."""
    resolved = _resolve(cwd, path)
    if not resolved.exists():
        return {"content": "", "error": f"File not found: {path}"}
    try:
        old_text = resolved.read_text(encoding="utf-8")
    except Exception:
        old_text = ""

    strip_val = str(strip if strip is not None else 0)

    async def _run_patch(patch_text: str) -> tuple[int, str, str]:
        try:
            with tempfile.NamedTemporaryFile("w+", delete=False) as tmp:
                tmp.write(patch_text)
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
            raise
        except Exception as exc:  # pragma: no cover - defensive
            return 1, "", str(exc)

        stdout_text = stdout.decode(errors="ignore") if stdout else ""
        stderr_text = stderr.decode(errors="ignore") if stderr else ""
        return proc.returncode or 0, stdout_text, stderr_text

    try:
        rc, stdout_text, stderr_text = await _run_patch(patch)
        if rc != 0:
            dedented = textwrap.dedent(patch)
            if dedented != patch:
                rc, stdout_text, stderr_text = await _run_patch(dedented)
        if rc != 0:
            error_msg = stderr_text or stdout_text or f"Patch failed ({rc})"
            return {"content": stdout_text, "error": error_msg}
        try:
            new_text = resolved.read_text(encoding="utf-8")
        except Exception:
            new_text = ""
        return {
            "content": stdout_text or "Patch applied",
            "error": None,
            "new_text": new_text,
            "old_text": old_text,
        }
    except FileNotFoundError:
        return {"content": "", "error": "`patch` command not found"}
