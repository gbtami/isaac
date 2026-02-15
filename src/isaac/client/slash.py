"""Client-side slash command registry and dispatch."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from acp import text_block

from isaac.client.acp_client import MODE_CONFIG_KEY, MODEL_CONFIG_KEY, set_config_option_value, set_mode
from isaac.client.session_state import SessionUIState
from isaac.client.status_box import build_status_banner
from isaac.client.thinking import toggle_thinking
from isaac.log_utils import log_event

logger = logging.getLogger(__name__)

SlashHandler = Callable[
    [Any, str, SessionUIState, Callable[[], None], str],
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


def _allowed_config_values(state: SessionUIState, key: str) -> list[str]:
    return sorted(state.config_option_values.get(key, set()))


def _group_models_by_provider(model_ids: list[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for model_id in model_ids:
        provider, sep, _rest = model_id.partition(":")
        group_key = provider if sep else "other"
        grouped.setdefault(group_key, []).append(model_id)
    for provider in grouped:
        grouped[provider] = sorted(grouped[provider])
    return dict(sorted(grouped.items()))


async def _pick_config_value(
    state: SessionUIState,
    *,
    key: str,
    label: str,
    current_value: str | None,
) -> str | None:
    options = _allowed_config_values(state, key)
    if not options:
        print(f"[agent did not advertise any {label} options]")
        return None
    selector = state.select_option
    if selector is None:
        print(f"[available {label} options: {', '.join(options)}]")
        return None
    return await selector(label, options, current_value)


@register_slash_command("/help", description="Show available slash commands.", hint="/help")
def _handle_help(
    _conn: Any,
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


@register_slash_command("/model", description="Set or pick model id.", hint="/model [id]")
async def _handle_model(
    conn: Any,
    session_id: str,
    state: SessionUIState,
    _permission_reset: Callable[[], None],
    argument: str,
) -> bool:
    allowed = _allowed_config_values(state, MODEL_CONFIG_KEY)
    providers = _group_models_by_provider(allowed)
    if argument:
        selection = argument.split()[0]
        if allowed and selection not in allowed:
            # Allow passing provider-only shorthand, then prompt model within provider.
            provider_models = providers.get(selection)
            if provider_models:
                selector = state.select_option
                if selector is None:
                    print(f"[available models for {selection}: {', '.join(provider_models)}]")
                    return True
                picked = await selector(
                    "model",
                    provider_models,
                    state.current_model if state.current_model in provider_models else provider_models[0],
                )
                if not picked:
                    return True
                selection = picked
            else:
                print(f"[unknown model: {selection}]")
                return True
    else:
        selector = state.select_option
        if selector is None:
            picked = await _pick_config_value(
                state,
                key=MODEL_CONFIG_KEY,
                label="model",
                current_value=state.current_model,
            )
            if not picked:
                return True
            selection = picked
        else:
            current_provider = (state.current_model or "").partition(":")[0] or None
            provider_ids = sorted(providers.keys())
            chosen_provider = (
                provider_ids[0]
                if len(provider_ids) == 1
                else await selector("provider", provider_ids, current_provider)
            )
            if not chosen_provider:
                return True
            provider_models = providers.get(chosen_provider)
            if not provider_models:
                print(f"[no models available for provider: {chosen_provider}]")
                return True
            picked = await selector(
                "model",
                provider_models,
                state.current_model if state.current_model in provider_models else provider_models[0],
            )
            if not picked:
                return True
            selection = picked

    if allowed and selection not in allowed:
        print(f"[unknown model: {selection}]")
        return True

    try:
        await set_config_option_value(conn, session_id, state, MODEL_CONFIG_KEY, selection)
        state.current_model = selection
        state.notify_changed()
        print(f"[model set to {selection}]")
        return True
    except Exception as exc:
        print(f"[failed to set model: {exc}]")
    return True


@register_slash_command("/thinking", description="Toggle display of model thinking traces.", hint="/thinking on|off")
def _handle_thinking(
    _conn: Any,
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
async def _handle_status(
    _conn: Any,
    _session_id: str,
    state: SessionUIState,
    _permission_reset: Callable[[], None],
    _argument: str,
) -> bool:
    previous_suppress = state.suppress_usage_output
    previous_deadline = state.suppress_usage_line_until
    previous_waiter = state.usage_refresh_waiter
    state.usage_refresh_waiter = asyncio.Event()
    state.suppress_usage_output = True
    state.suppress_usage_line_until = time.monotonic() + 1.0
    try:
        await _conn.prompt(prompt=[text_block("/usage")], session_id=_session_id)
        try:
            await asyncio.wait_for(state.usage_refresh_waiter.wait(), timeout=0.5)
        except asyncio.TimeoutError:
            pass
    except Exception as exc:
        log_event(logger, "client.status.refresh_usage.error", level=logging.DEBUG, error=str(exc))
    finally:
        state.usage_refresh_waiter = previous_waiter
        state.suppress_usage_output = previous_suppress
        state.suppress_usage_line_until = max(previous_deadline, state.suppress_usage_line_until)
    print(build_status_banner(state), end="")
    return True


@register_slash_command("/exit", description="Exit the client.", hint="/exit")
@register_slash_command("/quit", description="Exit the client.", hint="/quit")
def _handle_exit(
    _conn: Any,
    _session_id: str,
    _state: SessionUIState,
    _permission_reset: Callable[[], None],
    _argument: str,
) -> bool:
    print("[exiting]")
    raise SystemExit(0)


@register_slash_command("/mode", description="Set or pick agent mode.", hint="/mode [id]")
async def _handle_mode(
    conn: Any,
    session_id: str,
    state: SessionUIState,
    permission_reset: Callable[[], None],
    argument: str,
) -> bool:
    if argument:
        mode = argument.split()[0]
    else:
        picked = await _pick_config_value(
            state,
            key=MODE_CONFIG_KEY,
            label="mode",
            current_value=state.current_mode,
        )
        if not picked:
            return True
        mode = picked

    allowed = _allowed_config_values(state, MODE_CONFIG_KEY)
    if allowed and mode not in allowed:
        print(f"[unknown mode: {mode}]")
        return True
    await set_mode(conn, session_id, state, mode)
    permission_reset()
    return True


async def handle_slash_command(
    line: str,
    conn: Any,
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
            # Forward agent-advertised slash commands to the server.
            return False
        print(f"[unknown slash command: {command}]")
        _handle_help(conn, session_id, state, permission_reset, "")
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
