from __future__ import annotations

import asyncio
import os
import re
import signal
from pathlib import Path
from typing import Optional

from isaac.agent.ai_types import ToolContext
from isaac.agent.tools.safety import PathAccessError, ShellCommandDenied, resolve_command_cwd, validate_shell_command

DEFAULT_COMMAND_OUTPUT_CHARS = 20_000
MAX_COMMAND_OUTPUT_CHARS = 48_000
DEFAULT_COMMAND_TIMEOUT_S = 60.0
MAX_COMMAND_TIMEOUT_S = 300.0

_SECRET_ENV_NAME_PATTERNS = (
    r"TOKEN",
    r"SECRET",
    r"PASSWORD",
    r"PASSWD",
    r"PRIVATE",
    r"CREDENTIAL",
    r"AUTH",
    r"COOKIE",
    r"API[_-]?KEY",
    r"ACCESS[_-]?KEY",
)


def _bounded_output_chars(value: int | None) -> int:
    if value is None:
        return DEFAULT_COMMAND_OUTPUT_CHARS
    return max(1, min(int(value), MAX_COMMAND_OUTPUT_CHARS))


def _bounded_timeout(value: float | None) -> float:
    if value is None:
        return DEFAULT_COMMAND_TIMEOUT_S
    return max(0.1, min(float(value), MAX_COMMAND_TIMEOUT_S))


def _cap_output(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    marker = f"\n[truncated after {limit} characters]"
    keep = max(0, limit - len(marker))
    return text[:keep].rstrip() + marker, True


def _split_env_patterns(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for part in value.splitlines() for item in part.split(",") if item.strip()]


def _env_name_matches(name: str, patterns: tuple[str, ...] | list[str]) -> bool:
    for pattern in patterns:
        try:
            if re.search(pattern, name, re.IGNORECASE):
                return True
        except re.error:
            continue
    return False


def _command_env() -> tuple[dict[str, str], int]:
    """Return a command environment with common secret-like variables removed.

    Isaac is not a sandbox, but coding-agent commands should not inherit API
    tokens/passwords by default. The optional ``ISAAC_COMMAND_ENV_DENYLIST``
    accepts comma/newline-separated regular expressions for additional variable
    names to strip.
    """

    extra_deny = _split_env_patterns(os.getenv("ISAAC_COMMAND_ENV_DENYLIST"))
    env: dict[str, str] = {}
    stripped = 0
    for name, value in os.environ.items():
        if _env_name_matches(name, _SECRET_ENV_NAME_PATTERNS) or _env_name_matches(name, extra_deny):
            stripped += 1
            continue
        env[name] = value
    return env, stripped


def _terminate_process_group(proc: asyncio.subprocess.Process) -> None:
    """Kill a subprocess and its process group when supported."""

    pid = proc.pid
    if pid is None:  # pragma: no cover - defensive
        proc.kill()
        return
    if hasattr(os, "killpg"):
        try:
            os.killpg(pid, signal.SIGKILL)
            return
        except ProcessLookupError:
            return
        except OSError:
            pass
    proc.kill()


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
    timeout_s = _bounded_timeout(timeout)

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
            "timeout": timeout_s,
            "max_output_chars": output_limit,
        }

    command_env, stripped_env_count = _command_env()

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(resolved_cwd) if resolved_cwd else None,
            env=command_env,
            start_new_session=True,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        except asyncio.TimeoutError:
            _terminate_process_group(proc)
            stdout, stderr = await proc.communicate()
            stdout_text = stdout.decode(errors="replace").rstrip("\n") if stdout else ""
            stderr_text = stderr.decode(errors="replace").rstrip("\n") if stderr else ""
            stdout_capped, stdout_truncated = _cap_output(stdout_text, output_limit)
            stderr_capped, stderr_truncated = _cap_output(stderr_text, output_limit)
            return {
                "content": stdout_capped,
                "stderr": stderr_capped,
                "error": f"Command timed out after {timeout_s:g}s",
                "returncode": -1,
                "command": command,
                "cwd": str(resolved_cwd) if resolved_cwd else cwd,
                "timeout": timeout_s,
                "stdout_truncated": stdout_truncated,
                "stderr_truncated": stderr_truncated,
                "truncated": stdout_truncated or stderr_truncated,
                "max_output_chars": output_limit,
                "env_stripped_count": stripped_env_count,
            }
    except Exception as exc:
        return {
            "content": "",
            "stderr": "",
            "error": str(exc),
            "returncode": -1,
            "command": command,
            "cwd": str(resolved_cwd) if 'resolved_cwd' in locals() and resolved_cwd else cwd,
            "timeout": timeout_s,
            "max_output_chars": output_limit,
            "env_stripped_count": stripped_env_count if 'stripped_env_count' in locals() else 0,
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
        "timeout": timeout_s,
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
        "truncated": stdout_truncated or stderr_truncated,
        "max_output_chars": output_limit,
        "env_stripped_count": stripped_env_count,
    }
