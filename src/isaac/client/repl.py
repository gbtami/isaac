"""Interactive REPL loop and helpers for the ACP client."""

from __future__ import annotations

import asyncio
import getpass
import logging
import os
import sys
from pathlib import Path
from typing import Any

from acp import ClientSideConnection, text_block
from acp.schema import EmbeddedResourceContentBlock, ResourceContentBlock, TextResourceContents
from prompt_toolkit import PromptSession  # type: ignore
from prompt_toolkit.formatted_text import ANSI  # type: ignore
from prompt_toolkit.key_binding import KeyBindings  # type: ignore
from prompt_toolkit.patch_stdout import patch_stdout  # type: ignore
from prompt_toolkit.shortcuts import print_formatted_text  # type: ignore
from prompt_toolkit.styles import Style  # type: ignore

from isaac.client.display import create_thinking_status
from isaac.client.session_state import SessionUIState
from isaac.client.status_box import build_status_toolbar, build_welcome_banner, format_path
from isaac.client.slash import SLASH_HANDLERS, handle_slash_command
from isaac.log_utils import log_event

logger = logging.getLogger(__name__)

EMBED_FILE_MAX_BYTES = 20_000
PROMPT_STYLE = Style.from_dict(
    {
        "": "bg:#3a3a3a #ffffff",
        "prompt": "bg:#3a3a3a #ffffff",
        "prompt.symbol": "bg:#3a3a3a ansigreen",
        "prompt.user": "bg:#3a3a3a #ffffff",
        "prompt.sep": "bg:#3a3a3a #ffffff",
        "prompt.cwd": "bg:#3a3a3a #ffffff",
        "prompt.sig": "bg:#3a3a3a #ffffff",
    }
)


async def interactive_loop(conn: ClientSideConnection, session_id: str, state: SessionUIState) -> None:
    """Interactive REPL that drives session/prompt per ACP prompt turn rules."""
    kb = KeyBindings()
    CANCEL_TOKEN = "__CANCEL__"
    EXIT_TOKEN = "__EXIT__"

    @kb.add("escape")
    def _(event):  # type: ignore
        if not event.app.is_done:
            event.app.exit(result=CANCEL_TOKEN)

    @kb.add("c-q")
    def _(event):  # type: ignore
        if not event.app.is_done:
            event.app.exit(result=EXIT_TOKEN)

    session: PromptSession = PromptSession(
        key_bindings=kb,
        bottom_toolbar=lambda: build_status_toolbar(state),
        style=PROMPT_STYLE,
    )
    if state.thinking_status is None:
        state.thinking_status = create_thinking_status()

    def _invalidate_toolbar() -> None:
        try:
            session.app.invalidate()
        except Exception:
            return

    async def _select_option(
        label: str,
        options: list[str],
        current_value: str | None,
    ) -> str | None:
        if not options:
            return None
        default_value = current_value if current_value in options else options[0]
        print(f"Select {label}:")
        for idx, option in enumerate(options, start=1):
            marker = "*" if option == default_value else " "
            suffix = " (current)" if option == default_value else ""
            print(f"  {marker} {idx}. {option}{suffix}")

        while True:
            try:
                with patch_stdout():
                    selected = await session.prompt_async(
                        [
                            (
                                "class:prompt.sep",
                                f"{label} [1-{len(options)}] (Enter=current, q=cancel) $ ",
                            )
                        ],
                        default="",
                    )
            except (EOFError, KeyboardInterrupt):
                return None

            value = selected.strip()
            if not value:
                return default_value
            if value.lower() in {"q", "quit", "cancel"}:
                return None
            if value.isdigit():
                index = int(value)
                if 1 <= index <= len(options):
                    return options[index - 1]
            for option in options:
                if option.lower() == value.lower():
                    return option
            print(f"[invalid {label} selection: {value}]")

    state.refresh_ui = _invalidate_toolbar
    state.select_option = _select_option
    state.local_slash_commands = set(SLASH_HANDLERS.keys())
    if state.show_welcome_on_start:
        print(build_welcome_banner(state), end="")
        state.show_welcome_on_start = False

    while True:
        try:
            print_formatted_text(ANSI("\n"), end="")

            with patch_stdout():
                line = await session.prompt_async(_build_prompt(state))
            if line == CANCEL_TOKEN:
                state.cancel_requested = True
                await conn.cancel(session_id=session_id)
                print("[cancelled]")
                loop = asyncio.get_running_loop()
                loop.call_later(1.0, setattr, state, "cancel_requested", False)
                continue
            if line == EXIT_TOKEN:
                break
        except EOFError:
            break
        except KeyboardInterrupt:
            print("", file=sys.stderr)
            continue

        if not line:
            continue

        # Handle slash commands locally
        if line.startswith("/"):
            handled = await handle_slash_command(line, conn, session_id, state, permission_reset=lambda: None)
            if handled:
                continue

        blocks = _build_prompt_blocks(line)

        try:
            if state.thinking_status is not None:
                state.thinking_status.start()
            await conn.prompt(prompt=blocks, session_id=session_id)
        except Exception as exc:  # noqa: BLE001
            log_event(logger, "client.prompt.error", level=logging.WARNING, error=str(exc))
        finally:
            if state.thinking_status is not None:
                state.thinking_status.stop()

    # Unknown slash commands fall through to the agent for server-side handling.


def _build_prompt(state: SessionUIState) -> list[tuple[str, str]]:
    user = getpass.getuser()
    cwd = format_path(state.cwd)
    return [
        ("class:prompt.symbol", "🍏 "),
        ("class:prompt.user", user),
        ("class:prompt.sep", ":"),
        ("class:prompt.cwd", cwd),
        ("class:prompt.sig", " $ "),
    ]


def _build_prompt_blocks(line: str) -> list[Any]:
    """Build ACP content blocks from user input, embedding @file references."""

    blocks: list[Any] = [text_block(line)]
    refs = [word[1:] for word in line.split() if word.startswith("@") and len(word) > 1]
    for ref in refs:
        path = Path(ref)
        if not path.is_absolute():
            path = Path(os.getcwd()) / path
        if not path.exists() or not path.is_file():
            continue
        try:
            size = path.stat().st_size
        except Exception:
            size = 0
        uri = path.resolve().as_uri()
        if size <= EMBED_FILE_MAX_BYTES:  # embed small text files
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            res = TextResourceContents(text=text, uri=uri, mime_type="text/plain")
            blocks.append(EmbeddedResourceContentBlock(resource=res, type="resource"))
        else:
            blocks.append(
                ResourceContentBlock(
                    name=path.name,
                    uri=uri,
                    size=size,
                    mime_type="text/plain",
                    type="resource_link",
                )
            )
    return blocks
