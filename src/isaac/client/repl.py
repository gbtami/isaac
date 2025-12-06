"""Interactive REPL loop and helpers for the ACP client."""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable

from acp import ClientSideConnection, text_block
from prompt_toolkit import PromptSession  # type: ignore
from prompt_toolkit.key_binding import KeyBindings  # type: ignore

from isaac.client.acp_client import set_mode
from isaac.client.session_state import SessionUIState
from isaac.client.status_box import render_status_box
from isaac.client.thinking import toggle_thinking


async def interactive_loop(
    conn: ClientSideConnection, session_id: str, state: SessionUIState
) -> None:
    """Interactive REPL that drives session/prompt per ACP prompt turn rules."""
    kb = KeyBindings()
    CANCEL_TOKEN = "__CANCEL__"

    @kb.add("escape")
    def _(event):  # type: ignore
        if not event.app.is_done:
            event.app.exit(result=CANCEL_TOKEN)

    session: PromptSession = PromptSession(key_bindings=kb)
    if state.show_status_on_start:
        print(render_status_box(state))
        state.show_status_on_start = False

    while True:
        try:
            usage_suffix = f" [{state.usage_summary}]" if state.usage_summary else ""
            line = await session.prompt_async(
                f"{state.current_mode}|{state.current_model}{usage_suffix}> "
            )
            if line == CANCEL_TOKEN:
                await conn.cancel(session_id=session_id)
                print("[cancelled]")
                continue
        except EOFError:
            break
        except KeyboardInterrupt:
            print("", file=sys.stderr)
            continue

        if not line:
            continue

        # Handle slash commands locally
        if line.startswith("/"):
            handled = await _handle_slash(
                line, conn, session_id, state, permission_reset=lambda: None
            )
            if handled:
                continue
            # Guard against mistyped slash commands from going to the model.
            cmd = line.split()[0]
            agent_slash = {"/log", "/model", "/models"}
            if cmd not in agent_slash:
                print(
                    "Available slash commands: /status, /models, /model <id>, "
                    "/mode <ask|yolo>, /thinking on|off, /log <level>, /exit"
                )
                continue

        try:
            await conn.prompt(prompt=[text_block(line)], session_id=session_id)
        except Exception as exc:  # noqa: BLE001
            logging.error("Prompt failed: %s", exc)
        if state.pending_newline:
            print()
            state.pending_newline = False


async def _handle_slash(
    line: str,
    conn: ClientSideConnection,
    session_id: str,
    state: SessionUIState,
    permission_reset: Callable[[], None],
) -> bool:
    if line == "/help":
        print(
            "Available slash commands:\n"
            "/status       - Show current mode, model, and MCP servers\n"
            "/models       - List available models (handled by the agent)\n"
            "/model <id>   - Set model to the given id\n"
            "/mode <id>    - Set agent mode (ask|yolo)\n"
            "/thinking on|off - Toggle display of model thinking traces\n"
            "/log <level>  - Set agent log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)\n"
            "/exit         - Exit the client\n"
        )
        return True
    if line == "/status":
        print(render_status_box(state))
        return True
    if line.startswith("/model"):
        parts = line.split()
        if len(parts) == 1:
            print("Usage: /model <id> (use /models to list available models)")
            return True
        selection = parts[1]
        state.current_model = selection
        try:
            await conn.ext_method("model/set", {"session_id": session_id, "model_id": selection})
        except Exception:
            await conn.set_session_model(model_id=selection, session_id=session_id)
        print(f"[model set to {selection}]")
        return True
    if line.startswith("/thinking"):
        parts = line.split()
        if len(parts) == 2 and parts[1] in {"on", "off"}:
            msg = toggle_thinking(state, parts[1] == "on")
            print(msg)
        else:
            print("Usage: /thinking on|off")
        return True
    if line in ("/exit", "/quit"):
        print("[exiting]")
        raise SystemExit(0)

    if line.startswith("/mode"):
        parts = line.split()
        if len(parts) == 1:
            print("Usage: /mode <ask|yolo>")
            return True
        await set_mode(conn, session_id, state, parts[1])
        permission_reset()
        return True

    # Unknown slash commands fall through to the agent for server-side handling.
    return False
