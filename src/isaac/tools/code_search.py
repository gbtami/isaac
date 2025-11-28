from __future__ import annotations

import asyncio
import fnmatch
import re
from pathlib import Path
from typing import Optional


async def code_search(
    pattern: str,
    directory: str = ".",
    glob: Optional[str] = None,
    case_sensitive: bool = True,
    timeout: Optional[float] = None,
) -> dict:
    """Search for a pattern in code using ripgrep with a Python fallback."""
    path = Path(directory or ".")
    if not path.exists():
        return {
            "content": None,
            "error": f"Directory '{directory}' does not exist.",
            "returncode": -1,
        }
    if not path.is_dir():
        return {"content": None, "error": f"'{directory}' is not a directory.", "returncode": -1}

    cmd = ["rg", "--line-number", "--color", "never"]
    if not case_sensitive:
        cmd.append("-i")
    if glob:
        cmd.extend(["-g", glob])
    cmd.extend([pattern, str(path)])

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        stdout_text = stdout.decode() if stdout else ""
        stderr_text = stderr.decode() if stderr else ""
        error_text = stderr_text or None
        return {"content": stdout_text, "error": error_text, "returncode": proc.returncode}
    except FileNotFoundError:
        matches = []
        flags = 0 if case_sensitive else re.IGNORECASE
        regex = re.compile(pattern, flags)
        for file in path.rglob("*"):
            if not file.is_file():
                continue
            if glob and not fnmatch.fnmatch(file.name, glob):
                continue
            try:
                for idx, line in enumerate(file.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
                    if regex.search(line):
                        rel = file.relative_to(path)
                        matches.append(f"{rel}:{idx}:{line}")
            except Exception:
                continue
        return {"content": "\n".join(matches), "error": None, "returncode": 0}
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        return {"content": None, "error": f"Search timed out after {timeout}s", "returncode": -1}
    except Exception as exc:  # pragma: no cover - unexpected spawn errors
        return {"content": None, "error": str(exc), "returncode": -1}
