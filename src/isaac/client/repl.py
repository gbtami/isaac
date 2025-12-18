"""Interactive REPL loop and helpers for the ACP client."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

from acp import ClientSideConnection, text_block
from acp.schema import EmbeddedResourceContentBlock, ResourceContentBlock, TextResourceContents
from prompt_toolkit import PromptSession  # type: ignore
from prompt_toolkit.key_binding import KeyBindings  # type: ignore
from typing import Any

from isaac.client.session_state import SessionUIState
from isaac.client.status_box import render_status_box
from isaac.client.slash import handle_slash_command


async def interactive_loop(conn: ClientSideConnection, session_id: str, state: SessionUIState) -> None:
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
            if state.pending_newline:
                print()
                state.pending_newline = False
            usage_suffix = f" [{state.usage_summary}]" if state.usage_summary else ""
            line = await session.prompt_async(f"🍏 {state.current_mode}|{state.current_model}{usage_suffix}> ")
            if line == CANCEL_TOKEN:
                state.cancel_requested = True
                await conn.cancel(session_id=session_id)
                print("[cancelled]")
                loop = asyncio.get_running_loop()
                loop.call_later(1.0, setattr, state, "cancel_requested", False)
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
            handled = await handle_slash_command(line, conn, session_id, state, permission_reset=lambda: None)
            if handled:
                continue

        blocks = _build_prompt_blocks(line)

        try:
            await conn.prompt(prompt=blocks, session_id=session_id)
        except Exception as exc:  # noqa: BLE001
            logging.error("Prompt failed: %s", exc)
        if state.pending_newline:
            print()
            state.pending_newline = False

    # Unknown slash commands fall through to the agent for server-side handling.


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
        if size <= 20_000:  # embed small text files
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
