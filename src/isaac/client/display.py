"""Shared rich console utilities for client output."""

from __future__ import annotations

from typing import Iterable

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


def print_diff(text: str) -> None:
    """Render a unified diff with syntax highlighting."""
    console.print(Syntax(text, "diff", theme="ansi_dark", line_numbers=False))


def print_plan(entries: Iterable[str]) -> None:
    table = Table(title="Plan", show_header=False, box=None, border_style="cyan")
    table.add_column("#", width=4, style="cyan")
    table.add_column("Item", style="white")
    for idx, entry in enumerate(entries, start=1):
        table.add_row(str(idx), entry)
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
