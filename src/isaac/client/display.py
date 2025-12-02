"""Shared rich console utilities for client output."""

from __future__ import annotations

from typing import Iterable

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Use a rich console with colors enabled; prompt_toolkit handles stdout patching.
console = Console(force_terminal=True, color_system="standard", markup=False, highlight=False)


def print_mode_update(mode: str) -> None:
    console.print(Text(f"[mode -> {mode}]", style="magenta"))


def print_tool(status: str, message: str) -> None:
    style = "green" if status == "completed" else "yellow" if status == "in_progress" else "red"
    console.print(Text(f"| Tool[{status}]: {message}", style=style))


def print_agent_text(text: str) -> None:
    console.print(Text(text, style="cyan"), end="")


def print_thinking_text(text: str) -> None:
    console.print(Text(text, style="blue"), end="")


def print_plan(entries: Iterable[str]) -> None:
    table = Table(title="Plan", show_header=False, box=None, border_style="magenta")
    table.add_column("#", width=4, style="magenta")
    table.add_column("Item", style="white")
    for idx, entry in enumerate(entries, start=1):
        table.add_row(str(idx), entry)
    console.print(table)


def print_status(panel: Panel | str) -> None:
    if isinstance(panel, Panel):
        console.print(panel)
    else:
        console.print(panel)
