import asyncio
import contextvars
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from pydantic_ai import RunContext


@dataclass
class RunCommandContext:
    """Context for requesting approval before executing run_command."""

    request_permission: Callable[[str, Optional[str]], Awaitable[bool]]


_run_command_context: contextvars.ContextVar[RunCommandContext | None] = contextvars.ContextVar(
    "run_command_context", default=None
)


def set_run_command_context(ctx: RunCommandContext) -> contextvars.Token[RunCommandContext | None]:
    """Install a permission context for the current tool execution."""

    return _run_command_context.set(ctx)


def reset_run_command_context(token: contextvars.Token[RunCommandContext | None]) -> None:
    """Reset the permission context after the tool call finishes."""

    _run_command_context.reset(token)


def get_run_command_context() -> RunCommandContext | None:
    """Return the active run_command permission context, if any."""

    return _run_command_context.get()


async def run_command(
    ctx: RunContext[Any] = None,
    command: str = "",
    cwd: Optional[str] = None,
    timeout: Optional[float] = None,
) -> dict:
    """Execute a shell command and capture its output."""
    if command == "" and isinstance(ctx, str):
        command = ctx
        ctx = None
    ctx = get_run_command_context()
    if ctx:
        allowed = await ctx.request_permission(command, cwd)
        if not allowed:
            return {"content": "", "error": "permission denied", "returncode": -1}

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return {
                "content": "",
                "error": f"Command timed out after {timeout}s",
                "returncode": -1,
            }
    except Exception as exc:
        return {"content": "", "error": str(exc), "returncode": -1}

    stdout_text = stdout.decode().rstrip("\n") if stdout else ""
    stderr_text = stderr.decode().rstrip("\n") if stderr else ""
    error_text = stderr_text or None
    return {"content": stdout_text, "error": error_text, "returncode": proc.returncode}
