"""Client-side slash command registry and dispatch."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable

from acp import ClientSideConnection

from isaac.client.acp_client import set_mode
from isaac.client.session_state import SessionUIState
from isaac.client.thinking import toggle_thinking
from isaac.log_utils import log_event

logger = logging.getLogger(__name__)

SlashHandler = Callable[
    [ClientSideConnection, str, SessionUIState, Callable[[], None], str],
    Awaitable[bool] | bool,
]


@dataclass
class SlashCommandDef:
    description: str
    hint: str
    handler: SlashHandler


SLASH_HANDLERS: dict[str, SlashCommandDef] = {}


def register_slash_command(name: str, description: str, hint: str) -> Callable[[SlashHandler], SlashHandler]:
    """Decorator to register a slash command."""

    def _decorator(func: SlashHandler) -> SlashHandler:
        SLASH_HANDLERS[name] = SlashCommandDef(description=description, hint=hint, handler=func)
        return func

    return _decorator


@register_slash_command("/help", description="Show available slash commands.", hint="/help")
def _handle_help(
    _conn: ClientSideConnection,
    _session_id: str,
    _state: SessionUIState,
    _permission_reset: Callable[[], None],
    _argument: str,
) -> bool:
    print("Available slash commands:")
    for name, entry in SLASH_HANDLERS.items():
        print(f"{entry.hint:<16} - {entry.description}")
    if _state.available_agent_commands:
        for name, desc in sorted(_state.available_agent_commands.items()):
            if name in SLASH_HANDLERS:
                continue
            label = desc or "Handled by agent"
            print(f"{name:<16} - {label}")
    return True


@register_slash_command("/model", description="Set model to the given id.", hint="/model <id>")
async def _handle_model(
    conn: ClientSideConnection,
    session_id: str,
    state: SessionUIState,
    _permission_reset: Callable[[], None],
    argument: str,
) -> bool:
    if not argument:
        print("Usage: /model <id> (use /models to list available models)")
        return True
    selection = argument.split()[0]
    try:
        resp = await conn.ext_method("model/set", {"session_id": session_id, "model_id": selection})
        if isinstance(resp, dict) and resp.get("error"):
            print(f"[failed to set model: {resp['error']}]")
            return True
        state.current_model = selection
        state.notify_changed()
        print(f"[model set to {selection}]")
        return True
    except Exception as exc:
        try:
            await conn.set_session_model(model_id=selection, session_id=session_id)
            state.current_model = selection
            state.notify_changed()
            print(f"[model set to {selection}]")
        except Exception as inner_exc:  # noqa: BLE001
            print(f"[failed to set model: {inner_exc or exc}]")
    return True


@register_slash_command("/thinking", description="Toggle display of model thinking traces.", hint="/thinking on|off")
def _handle_thinking(
    _conn: ClientSideConnection,
    _session_id: str,
    state: SessionUIState,
    _permission_reset: Callable[[], None],
    argument: str,
) -> bool:
    parts = argument.split()
    if len(parts) == 1 and parts[0] in {"on", "off"}:
        msg = toggle_thinking(state, parts[0] == "on")
        print(msg)
    else:
        print("Usage: /thinking on|off")
    return True


@register_slash_command("/status", description="Show current session status.", hint="/status")
def _handle_status(
    _conn: ClientSideConnection,
    _session_id: str,
    state: SessionUIState,
    _permission_reset: Callable[[], None],
    _argument: str,
) -> bool:
    print(f"Session: {state.session_id or 'unknown'}")
    print(f"Mode: {state.current_mode or 'unknown'}")
    print(f"Model: {state.current_model or 'unknown'}")
    if state.cwd:
        print(f"Cwd: {state.cwd}")
    if state.mcp_servers:
        print(f"MCP: {', '.join(state.mcp_servers)}")
    if state.usage_summary:
        print(state.usage_summary)
    return True


@register_slash_command("/exit", description="Exit the client.", hint="/exit")
@register_slash_command("/quit", description="Exit the client.", hint="/quit")
def _handle_exit(
    _conn: ClientSideConnection,
    _session_id: str,
    _state: SessionUIState,
    _permission_reset: Callable[[], None],
    _argument: str,
) -> bool:
    print("[exiting]")
    raise SystemExit(0)


@register_slash_command("/mode", description="Set agent mode (ask|yolo).", hint="/mode <ask|yolo>")
async def _handle_mode(
    conn: ClientSideConnection,
    session_id: str,
    state: SessionUIState,
    permission_reset: Callable[[], None],
    argument: str,
) -> bool:
    parts = argument.split()
    if not parts:
        print("Usage: /mode <ask|yolo>")
        return True
    await set_mode(conn, session_id, state, parts[0])
    permission_reset()
    return True


async def handle_slash_command(
    line: str,
    conn: ClientSideConnection,
    session_id: str,
    state: SessionUIState,
    permission_reset: Callable[[], None],
) -> bool:
    """Dispatch client-side slash commands, returning True if handled."""
    trimmed = line.strip()
    if not trimmed.startswith("/"):
        return False

    parts = trimmed.split(maxsplit=1)
    command = parts[0] if parts else ""
    argument = parts[1].strip() if len(parts) > 1 else ""

    entry = SLASH_HANDLERS.get(command)
    if entry is None:
        if command in state.available_agent_commands:
            return False
        print(f"[unknown command: {command} (try /help)]")
        return True

    try:
        result = entry.handler(conn, session_id, state, permission_reset, argument)
        if asyncio.iscoroutine(result):
            return bool(await result)
        return bool(result)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        log_event(logger, "client.slash.error", level=logging.WARNING, command=command, error=str(exc))
        return True
