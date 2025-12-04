"""Slash command helpers mapped to ACP slash-commands.

See: https://agentclientprotocol.com/protocol/slash-commands
"""

from __future__ import annotations

import logging
from acp import SessionNotification
from acp.helpers import update_agent_message, text_block


def handle_slash_command(session_id: str, prompt: str) -> SessionNotification | None:
    """Handle server-side slash commands (Slash Commands section)."""
    trimmed = prompt.strip()
    parts = trimmed.split(maxsplit=1)
    command = parts[0] if parts else ""

    if command == "/log":
        level = parts[1] if len(parts) > 1 else ""
        output = _set_log_level(level)
        return SessionNotification(
            sessionId=session_id,
            update=update_agent_message(text_block(output)),
        )
    return None


def _set_log_level(level: str) -> str:
    """Set logging level for the main agent and provider loggers."""
    level_name = (level or "").strip().upper()
    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

    if not level_name:
        return _log_usage(valid_levels)
    if level_name not in valid_levels:
        return _log_usage(valid_levels, invalid=level)

    target_level = getattr(logging, level_name, logging.INFO)
    logging.getLogger().setLevel(target_level)
    for name in (
        "acp_server",
        "pydantic_ai",
        "pydantic_ai.providers",
        "httpx",
    ):
        logging.getLogger(name).setLevel(target_level)
    return f"Logging level set to {level_name}. Future requests will include verbose provider/LLM events."


def _log_usage(valid_levels: set[str], invalid: str | None = None) -> str:
    msg = ""
    if invalid is not None:
        msg = f"Unsupported log level '{invalid}'. "
    current = logging.getLogger().getEffectiveLevel()
    current_name = logging.getLevelName(current)
    return (
        f"{msg}Usage: /log <level>. "
        f"Valid levels: {', '.join(sorted(valid_levels))}. Current: {current_name}."
    )
