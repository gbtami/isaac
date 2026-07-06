from __future__ import annotations

import asyncio
import fnmatch
import re
from pathlib import Path
from typing import Optional

from isaac.agent.ai_types import ToolContext
from isaac.agent.tools.safety import BinaryFileError, PathAccessError, ensure_text_file, resolve_workspace_path

DEFAULT_CODE_SEARCH_MAX_RESULTS = 100
MAX_CODE_SEARCH_RESULTS = 500


def _bounded_max_results(value: int | None) -> int:
    if value is None:
        return DEFAULT_CODE_SEARCH_MAX_RESULTS
    return max(1, min(int(value), MAX_CODE_SEARCH_RESULTS))


def _cap_matches(matches: list[str], max_results: int) -> tuple[str, bool, int]:
    shown = matches[:max_results]
    truncated = len(matches) > len(shown)
    content = "\n".join(shown)
    if truncated:
        content = f"{content}\n[truncated after {len(shown)} of {len(matches)} matches]"
    return content, truncated, len(matches)


async def code_search(
    ctx: ToolContext | None = None,
    pattern: str = "",
    directory: str = ".",
    glob: Optional[str] = None,
    case_sensitive: bool = True,
    timeout: Optional[float] = None,
    max_results: Optional[int] = None,
    cwd: Optional[str] = None,
    session_cwd: str | Path | None = None,
    additional_directories: tuple[str | Path, ...] = (),
) -> dict:
    """Search for a pattern in code using ripgrep with a bounded result set."""

    _ = ctx
    result_limit = _bounded_max_results(max_results)
    try:
        path = resolve_workspace_path(session_cwd or cwd, directory, additional_directories=additional_directories)
    except PathAccessError as exc:
        return {
            "content": None,
            "error": str(exc),
            "returncode": -1,
        }
    if not path.exists():
        return {
            "content": None,
            "error": f"Directory '{directory}' does not exist.",
            "returncode": -1,
        }
    if not path.is_dir():
        return {"content": None, "error": f"'{directory}' is not a directory.", "returncode": -1}

    cmd = ["rg", "--line-number", "--color", "never", "--no-messages"]
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
        matches = [line for line in stdout_text.splitlines() if line]
        content, truncated, match_count = _cap_matches(matches, result_limit)
        error_text = stderr_text or None
        # ripgrep returns 1 for "no matches"; that is a completed search, not a failed tool call.
        returncode = 0 if proc.returncode == 1 and not error_text else proc.returncode
        return {
            "content": content,
            "error": error_text,
            "returncode": returncode,
            "match_count": match_count,
            "shown_count": min(match_count, result_limit),
            "max_results": result_limit,
            "truncated": truncated,
            "pattern": pattern,
            "directory": directory,
        }
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
                ensure_text_file(file)
                for idx, line in enumerate(file.read_text(encoding="utf-8").splitlines(), start=1):
                    if regex.search(line):
                        rel = file.relative_to(path)
                        matches.append(f"{rel}:{idx}:{line}")
            except (BinaryFileError, UnicodeDecodeError, OSError):
                continue
        content, truncated, match_count = _cap_matches(matches, result_limit)
        return {
            "content": content,
            "error": None,
            "returncode": 0,
            "match_count": match_count,
            "shown_count": min(match_count, result_limit),
            "max_results": result_limit,
            "truncated": truncated,
            "pattern": pattern,
            "directory": directory,
        }
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        return {"content": None, "error": f"Search timed out after {timeout}s", "returncode": -1}
    except Exception as exc:  # pragma: no cover - unexpected spawn errors
        return {"content": None, "error": str(exc), "returncode": -1}
