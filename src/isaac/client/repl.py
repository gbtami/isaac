"""Interactive REPL loop and helpers for the ACP client."""

from __future__ import annotations

import asyncio
import logging
import sys
from collections.abc import Callable

from acp import CancelNotification, ClientSideConnection, PromptRequest, text_block
from acp import SetSessionModelRequest
from prompt_toolkit import PromptSession  # type: ignore
from prompt_toolkit.key_binding import KeyBindings  # type: ignore

from isaac.client.protocol import set_mode
from isaac.client.session_state import SessionUIState
from isaac.client.status_box import render_status_box
from isaac.client.thinking import toggle_thinking


async def read_console(prompt: str) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: input(prompt))


async def _run_tests() -> int:
    proc = await asyncio.create_subprocess_shell(
        "uv run pytest",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    assert proc.stdout is not None
    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        print(line.decode().rstrip())
    return await proc.wait()


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

    session = PromptSession(key_bindings=kb)
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
                await conn.cancel(CancelNotification(sessionId=session_id))
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

        try:
            await conn.prompt(
                PromptRequest(
                    sessionId=session_id,
                    prompt=[text_block(line)],
                )
            )
        except Exception as exc:  # noqa: BLE001
            logging.error("Prompt failed: %s", exc)
        if state.pending_newline:
            print()
            state.pending_newline = False


async def _interactive_model_select(
    conn: ClientSideConnection, session_id: str, state: SessionUIState
) -> None:
    """Ask the agent for available models and allow user to pick one."""
    # Prefer extension method if supported
    choices: list[str] = []
    try:
        resp = await conn.extMethod("model/list", {"sessionId": session_id})
        if isinstance(resp, dict):
            current = resp.get("current")
            models = resp.get("models") or []
            if isinstance(current, str):
                state.current_model = current
            choices = [m.get("id") for m in models if isinstance(m, dict) and m.get("id")]
    except Exception:
        choices = []

    if not choices:
        # Fallback: ask via prompt and parse response
        state.collect_models = True
        state.model_buffer = []
        try:
            await conn.prompt(
                PromptRequest(
                    sessionId=session_id,
                    prompt=[text_block("/model")],
                )
            )
        except Exception as exc:  # noqa: BLE001
            logging.error("Failed to fetch models: %s", exc)
            state.collect_models = False
            return
        state.collect_models = False
        text = "\n".join(state.model_buffer or [])
        choices = _parse_models_from_text(text)

    if not choices:
        print("[no models returned by agent]")
        return

    print("Available models:")
    for idx, mid in enumerate(choices, start=1):
        print(f"{idx}) {mid}")
    loop = asyncio.get_running_loop()
    try:
        raw = await loop.run_in_executor(None, lambda: input("Select model (number): ").strip())
    except Exception:
        return
    if not raw.isdigit():
        return
    num = int(raw)
    if 1 <= num <= len(choices):
        selection = choices[num - 1]
        state.current_model = selection
        try:
            # Use extension method first, fall back to core set_model if needed.
            try:
                await conn.extMethod("model/set", {"sessionId": session_id, "modelId": selection})
            except Exception:
                await conn.setSessionModel(
                    SetSessionModelRequest(sessionId=session_id, modelId=selection)
                )
            print(f"[model set to {selection}]")
        except Exception as exc:  # noqa: BLE001
            logging.error("Failed to set model: %s", exc)


def _parse_models_from_text(text: str) -> list[str]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    model_ids: list[str] = []
    for line in lines:
        if line.startswith("-"):
            part = line.lstrip("-").strip()
            if part:
                model_ids.append(part.split()[0])
        elif ":" in line and line.lower().startswith("current model"):
            # skip current model line
            continue
        elif line and " " not in line and line != "Available":
            model_ids.append(line)
    return model_ids


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
            "/models       - Fetch available models from the agent and select one\n"
            "/model <id>   - Set model to the given id\n"
            "/mode <id>    - Set agent mode (ask|yolo)\n"
            "/thinking on|off - Toggle display of model thinking traces\n"
            "/test         - Run pytest locally\n"
            "/exit         - Exit the client\n"
        )
        return True
    if line == "/models":
        await _interactive_model_select(conn, session_id, state)
        return True
    if line == "/status":
        print(render_status_box(state))
        return True
    if line.startswith("/model"):
        parts = line.split()
        if len(parts) == 1:
            await _interactive_model_select(conn, session_id, state)
            return True
        selection = parts[1]
        state.current_model = selection
        try:
            await conn.extMethod("model/set", {"sessionId": session_id, "modelId": selection})
        except Exception:
            await conn.setSessionModel(
                SetSessionModelRequest(sessionId=session_id, modelId=selection)
            )
        print(f"[model set to {selection}]")
        return True
    if line.startswith("/mode"):
        parts = line.split()
        if len(parts) == 1:
            print("Usage: /mode <ask|yolo>")
            return True
        await set_mode(conn, session_id, state, parts[1])
        permission_reset()
        return True
    if line == "/test":
        print("[running tests: uv run pytest]")
        await _run_tests()
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

    print(
        "Available slash commands: /status, /models, /model <id>, /mode <ask|yolo>, "
        "/thinking on|off, /test, /exit"
    )
    return True
