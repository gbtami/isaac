"""Shared rich console utilities for client output."""

from __future__ import annotations

from typing import Any, Iterable
import ast
import contextlib

from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
import difflib

# Use a rich console with colors enabled; prompt_toolkit handles stdout patching.
console = Console(force_terminal=True, color_system="standard", markup=False, highlight=False)


def print_mode_update(mode: str) -> None:
    console.print(Text(f"[mode -> {mode}]", style="magenta"))


def print_tool(status: str, message: str) -> None:
    normalized = status.lower()
    style = (
        "green"
        if normalized == "completed"
        else "yellow"
        if normalized in {"in_progress", "start"}
        else "red"
    )
    console.print(Text(f"| Tool[{status}]: {message}", style=style))


def print_agent_text(text: str) -> None:
    console.print(Text(text, style="green"), end="")


def print_thought(text: str) -> None:
    console.print(Text(text, style="cyan dim"), end="")


def print_diff(text: str) -> None:
    """Render a unified diff with syntax highlighting."""
    console.print(Syntax(text, "diff", theme="ansi_dark", line_numbers=False))


def print_plan(entries: Iterable[Any]) -> None:
    """Render plan entries with status/priority."""
    table = Table(show_header=False, box=None, border_style="cyan")
    table.add_column("", width=2, style="cyan")
    table.add_column("Priority", width=9, style="yellow")
    table.add_column("Item", style="white")
    status_icons = {
        "completed": ("[✓]", "green"),
        "in_progress": ("[~]", "yellow"),
        "pending": ("[ ]", "cyan"),
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
                    return "\n".join(
                        f"- {str(item).strip()}" for item in parsed if str(item).strip()
                    )
        return text

    for entry in entries:
        status = getattr(entry, "status", "pending") or "pending"
        icon, color = status_icons.get(status, ("[ ]", "cyan"))
        priority = getattr(entry, "priority", "medium") or "medium"
        content = _format_content(getattr(entry, "content", "") or "")
        table.add_row(Text(icon, style=color), priority, content)
    console.print(table)


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
