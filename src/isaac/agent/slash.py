"""Slash command helpers mapped to ACP slash-commands.

See: https://agentclientprotocol.com/protocol/slash-commands
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from acp import SessionNotification
from acp.helpers import session_notification, update_agent_message, text_block
from acp.schema import AvailableCommand, AvailableCommandInput, UnstructuredCommandInput

from isaac.agent import models as model_registry
from isaac.agent.usage import format_usage_summary, normalize_usage

SLASH_HANDLERS: dict[str, "SlashCommandDef"] = {}

SlashHandler = Callable[
    [Any, str, str, str], Awaitable[SessionNotification | None] | SessionNotification | None
]


@dataclass
class SlashCommandDef:
    description: str
    hint: str
    handler: SlashHandler


def register_slash_command(
    name: str, description: str, hint: str
) -> Callable[[SlashHandler], SlashHandler]:
    """Decorator to register a slash command handler."""

    def _decorator(func: SlashHandler) -> SlashHandler:
        SLASH_HANDLERS[name] = SlashCommandDef(description=description, hint=hint, handler=func)
        return func

    return _decorator


def _current_model(agent: Any, session_id: str) -> str:
    session_models = getattr(agent, "_session_model_ids", {}) or {}
    if session_id in session_models:
        return session_models.get(session_id, "") or ""
    getter = getattr(agent, "_current_model_id", None)
    if callable(getter):
        return getter()
    return ""


def _set_current_model(agent: Any, session_id: str, model_id: str) -> None:
    try:
        session_models = getattr(agent, "_session_model_ids", None)
        if isinstance(session_models, dict):
            session_models[session_id] = model_id
    except Exception:
        return


def _list_models(agent: Any, session_id: str) -> SessionNotification:
    current = _current_model(agent, session_id)
    models = model_registry.list_user_models()
    lines = [f"Current model: {current}"]
    if models:
        lines.append("Available models:")
        for model_id, meta in models.items():
            desc = meta.get("description") or ""
            prefix = "*" if model_id == current else "-"
            line = f"{prefix} {model_id}"
            if desc:
                line = f"{line} - {desc}"
            lines.append(line)
    else:
        lines.append("No models configured.")
    return session_notification(
        session_id,
        update_agent_message(text_block("\n".join(lines))),
    )


@register_slash_command(
    "/model",
    description="Switch to a specific model.",
    hint="/model <id>",
)
async def _handle_model(agent: Any, session_id: str, _: str, argument: str) -> SessionNotification:
    if not argument:
        return _list_models(agent, session_id)

    models = model_registry.list_user_models()
    if argument not in models:
        message = f"Unknown model id: {argument}"
        return session_notification(session_id, update_agent_message(text_block(message)))

    try:
        await agent.set_session_model(argument, session_id)
        _set_current_model(agent, session_id, argument)
        message = f"Model set to {argument}."
    except Exception as exc:  # noqa: BLE001
        message = f"Failed to set model: {exc}"
    return session_notification(session_id, update_agent_message(text_block(message)))


@register_slash_command(
    "/models",
    description="List available models.",
    hint="/models",
)
def _handle_models(
    agent: Any, session_id: str, _command: str, _argument: str
) -> SessionNotification:
    # /models and bare /model both list available models.
    return _list_models(agent, session_id)


@register_slash_command(
    "/log",
    description="Set agent log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    hint="/log <level>",
)
def _handle_log(agent: Any, session_id: str, _command: str, argument: str) -> SessionNotification:
    output = _set_log_level(argument)
    return SessionNotification(
        session_id=session_id,
        update=update_agent_message(text_block(output)),
    )


@register_slash_command(
    "/usage",
    description="Show token usage for the latest run.",
    hint="/usage",
)
def _handle_usage(
    agent: Any, session_id: str, _command: str, _argument: str
) -> SessionNotification:
    try:
        # Prefer existing helper when available to keep formatting consistent.
        builder = getattr(agent, "_build_usage_note", None)
        if callable(builder):
            return builder(session_id)
    except Exception:
        pass

    model_id = getattr(agent, "_session_model_ids", {}).get(session_id, "")  # type: ignore[index]
    context_limit = model_registry.get_context_limit(model_id)
    usage = normalize_usage(getattr(agent, "_session_usage", {}).get(session_id))  # type: ignore[index]
    summary = format_usage_summary(usage, context_limit, model_id)
    return session_notification(session_id, update_agent_message(text_block(summary)))


@register_slash_command(
    "/strategies",
    description="List available prompt strategies.",
    hint="/strategies",
)
def _handle_strategies(
    agent: Any, session_id: str, _command: str, _argument: str
) -> SessionNotification:
    describer = getattr(agent, "describe_prompt_strategies", None)
    if callable(describer):
        message = describer(session_id)
    else:  # pragma: no cover - defensive fallback
        message = "Prompt strategies are not supported by this agent."
    return session_notification(session_id, update_agent_message(text_block(message)))


@register_slash_command(
    "/strategy",
    description="Set the prompt strategy (handoff|delegation|single|plan_only).",
    hint="/strategy <id>",
)
async def _handle_strategy(
    agent: Any, session_id: str, _command: str, argument: str
) -> SessionNotification:
    setter = getattr(agent, "set_session_prompt_strategy", None)
    if not callable(setter):
        message = "Prompt strategy selection is not supported by this agent."
    else:
        result = setter(argument, session_id)
        message = await result if asyncio.iscoroutine(result) else result
    return session_notification(session_id, update_agent_message(text_block(message)))


async def handle_slash_command(
    agent: Any, session_id: str, prompt: str
) -> SessionNotification | None:
    """Handle server-side slash commands (Slash Commands section)."""
    trimmed = prompt.strip()
    if not trimmed.startswith("/"):
        return None

    parts = trimmed.split(maxsplit=1)
    command = parts[0] if parts else ""
    argument = parts[1].strip() if len(parts) > 1 else ""

    entry = SLASH_HANDLERS.get(command)
    if entry is None:
        return None

    result = entry.handler(agent, session_id, command, argument)
    if asyncio.iscoroutine(result):
        return await result
    return result


def available_slash_commands() -> list[AvailableCommand]:
    """Build ACP AvailableCommand entries from registered slash commands."""
    commands: list[AvailableCommand] = []
    for name, entry in SLASH_HANDLERS.items():
        commands.append(
            AvailableCommand(
                name=name.lstrip("/"),
                description=entry.description,
                input=AvailableCommandInput(root=UnstructuredCommandInput(hint=entry.hint)),
            )
        )
    return commands


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
        "isaac.llm",
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
