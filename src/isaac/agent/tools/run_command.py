import asyncio
from typing import Optional

from isaac.agent.ai_types import ToolContext


async def run_command(
    ctx: ToolContext | None = None,
    command: str = "",
    cwd: Optional[str] = None,
    timeout: Optional[float] = None,
) -> dict:
    """Execute a shell command and capture its output."""
    if command == "" and isinstance(ctx, str):
        command = ctx
        ctx = None

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
