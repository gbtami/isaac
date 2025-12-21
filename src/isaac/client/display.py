"""Shared rich console utilities for client output."""

from __future__ import annotations

from typing import Any, Iterable
import ast
import contextlib
from io import StringIO
from threading import Lock

from prompt_toolkit.formatted_text import ANSI  # type: ignore
from prompt_toolkit.shortcuts import print_formatted_text  # type: ignore
from rich.console import Console
from rich.status import Status
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
import difflib

# Use a rich console with colors enabled; prompt_toolkit handles stdout patching.
_status_console = Console(force_terminal=True, color_system="standard", markup=False, highlight=False)
_render_buffer = StringIO()
_render_console = Console(
    file=_render_buffer,
    force_terminal=True,
    color_system="standard",
    markup=False,
    highlight=False,
)
_render_lock = Lock()


class ThinkingStatus:
    def __init__(self, console: Console) -> None:
        self._console = console
        self._lock = Lock()
        self._status: Status | None = None

    def start(self, text: str = "Thinking...") -> None:
        with self._lock:
            if self._status is not None:
                return
            self._status = self._console.status(Text(text, style="cyan"))
            self._status.start()

    def stop(self) -> None:
        with self._lock:
            if self._status is None:
                return
            self._status.stop()
            self._status = None


def create_thinking_status() -> ThinkingStatus:
    return ThinkingStatus(_status_console)


def _render_and_print(*args: Any, **kwargs: Any) -> bool:
    end = kwargs.get("end")
    if end is None:
        kwargs["end"] = "\n"
    with _render_lock:
        _render_buffer.seek(0)
        _render_buffer.truncate(0)
        _render_console.print(*args, **kwargs)
        output = _render_buffer.getvalue()
    if output:
        print_formatted_text(ANSI(output), end="")
        return output.endswith("\n")
    return False


def print_mode_update(mode: str) -> None:
    _render_and_print(Text(f"[mode -> {mode}]", style="magenta"))


def print_tool(status: str, message: str, *, kind: str | None = None) -> None:
    normalized = status.lower()
    style = "green" if normalized == "completed" else "yellow" if normalized in {"in_progress", "start"} else "red"
    _render_and_print(Text(f"🛠️ | Tool[{status}]: {message}", style=style))


def _render_text(text: str, style: str | None) -> Text:
    if "\x1b" in text:
        return Text.from_ansi(text)
    if style:
        return Text(text, style=style)
    return Text(text)


def print_agent_text(text: str) -> None:
    _render_and_print(_render_text(text, None), end="")


def print_thought(text: str) -> None:
    _render_and_print(_render_text(text, "#aaaaaa"), end="")


def print_diff(text: str) -> None:
    """Render a unified diff with syntax highlighting."""
    _render_and_print(Syntax(text, "diff", theme="ansi_dark", line_numbers=False))


def print_plan(entries: Iterable[Any]) -> None:
    """Render plan entries with a status dot."""
    table = Table(show_header=False, box=None, border_style="cyan")
    table.add_column("", width=2, style="cyan")
    table.add_column("Item", style="white")
    status_styles = {
        "completed": "green",
        "in_progress": "orange1",
        "pending": "orange1",
    }

    def _format_content(raw: Any) -> str:
        if not isinstance(raw, str):
            return str(raw)
        text = raw.strip()
        if text.startswith("steps="):
            text = text.split("=", 1)[1].strip()
        if text.startswith("[") and text.endswith("]"):
            with contextlib.suppress(Exception):
                parsed = ast.literal_eval(text)
                if isinstance(parsed, list):
                    return "\n".join(f"- {str(item).strip()}" for item in parsed if str(item).strip())
        return text

    for entry in entries:
        status = getattr(entry, "status", "pending") or "pending"
        style = status_styles.get(status, "orange1")
        content = _format_content(getattr(entry, "content", "") or "")
        table.add_row(Text("•", style=style), content)
    _render_and_print(table)


def print_file_edit_diff(path: str, old_text: str | None, new_text: str) -> None:
    """Render a file edit as a unified diff."""
    old_lines = (old_text or "").splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    diff = "".join(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=path or "before",
            tofile=path or "after",
            lineterm="",
        )
    )
    print_diff(diff if diff else f"No changes for {path or '<file>'}")
