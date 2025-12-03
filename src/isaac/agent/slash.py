"""Slash command helpers mapped to ACP slash-commands.

See: https://agentclientprotocol.com/protocol/slash-commands
"""

from __future__ import annotations

import subprocess
from acp import SessionNotification
from acp.helpers import update_agent_message, text_block


def handle_slash_command(session_id: str, prompt: str) -> SessionNotification | None:
    """Handle server-side slash commands (Slash Commands section)."""
    if prompt.strip() == "/test":
        output = _run_pytest()
        return SessionNotification(
            sessionId=session_id,
            update=update_agent_message(text_block(output)),
        )
    return None


def _run_pytest() -> str:
    try:
        proc = subprocess.run(
            ["uv", "run", "pytest"],
            capture_output=True,
            text=True,
            check=False,
        )
        status = "ok" if proc.returncode == 0 else f"failed ({proc.returncode})"
        return f"Tests {status}.\n{proc.stdout or ''}{proc.stderr or ''}".strip()
    except Exception as exc:  # pragma: no cover - subprocess failures
        return f"Failed to run tests: {exc}"
