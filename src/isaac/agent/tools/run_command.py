from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from isaac.agent.ai_types import ToolContext
from isaac.agent.tools.safety import PathAccessError, ShellCommandDenied, resolve_command_cwd, validate_shell_command

DEFAULT_COMMAND_OUTPUT_CHARS = 20_000
MAX_COMMAND_OUTPUT_CHARS = 48_000


def _bounded_output_chars(value: int | None) -> int:
    if value is None:
        return DEFAULT_COMMAND_OUTPUT_CHARS
    return max(1, min(int(value), MAX_COMMAND_OUTPUT_CHARS))


def _cap_output(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    marker = f"\n[truncated after {limit} characters]"
    keep = max(0, limit - len(marker))
    return text[:keep].rstrip() + marker, True


async def run_command(
    ctx: ToolContext | None = None,
    command: str = "",
    cwd: Optional[str] = None,
    timeout: Optional[float] = None,
    max_output_chars: Optional[int] = None,
    session_cwd: str | Path | None = None,
    additional_directories: tuple[str | Path, ...] = (),
) -> dict:
    """Execute a shell command and return bounded stdout/stderr diagnostics."""

    if command == "" and isinstance(ctx, str):
        command = ctx
        ctx = None
    _ = ctx
    output_limit = _bounded_output_chars(max_output_chars)

    try:
        validate_shell_command(command)
        resolved_cwd = resolve_command_cwd(
            session_cwd,
            cwd or ".",
            additional_directories=additional_directories,
        )
    except (ShellCommandDenied, PathAccessError) as exc:
        return {
            "content": "",
            "stderr": "",
            "error": str(exc),
            "returncode": -1,
            "command": command,
            "cwd": cwd,
            "max_output_chars": output_limit,
        }

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(resolved_cwd) if resolved_cwd else None,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return {
                "content": "",
                "stderr": "",
                "error": f"Command timed out after {timeout}s",
                "returncode": -1,
                "command": command,
                "cwd": str(resolved_cwd) if resolved_cwd else cwd,
                "max_output_chars": output_limit,
            }
    except Exception as exc:
        return {
            "content": "",
            "stderr": "",
            "error": str(exc),
            "returncode": -1,
            "command": command,
            "cwd": str(resolved_cwd) if 'resolved_cwd' in locals() and resolved_cwd else cwd,
            "max_output_chars": output_limit,
        }

    stdout_text = stdout.decode(errors="replace").rstrip("\n") if stdout else ""
    stderr_text = stderr.decode(errors="replace").rstrip("\n") if stderr else ""
    stdout_capped, stdout_truncated = _cap_output(stdout_text, output_limit)
    stderr_capped, stderr_truncated = _cap_output(stderr_text, output_limit)
    returncode = proc.returncode
    error_text = stderr_capped if returncode not in (0, None) and stderr_capped else None
    return {
        "content": stdout_capped,
        "stderr": stderr_capped,
        "error": error_text,
        "returncode": returncode,
        "command": command,
        "cwd": str(resolved_cwd) if resolved_cwd else cwd,
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
        "truncated": stdout_truncated or stderr_truncated,
        "max_output_chars": output_limit,
    }
