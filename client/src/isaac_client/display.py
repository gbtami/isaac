"""Rich output and prompt-toolkit menus for the terminal client."""

import asyncio
import difflib
import sys
from collections.abc import Callable

from acp import schema
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text


def visible(text: str) -> str:
    """Render untrusted ACP labels and chunks without terminal control sequences."""
    return "".join(
        character
        for character in text
        if character in "\t\n" or ord(character) >= 32 and not 127 <= ord(character) < 160
    )


class Terminal:
    def __init__(self) -> None:
        self.console = Console(markup=False, highlight=False)
        self.interactive = sys.stdin.isatty()
        self._prompt: PromptSession[str] | None = None
        self._menu_lock = asyncio.Lock()

    def write(self, text: str, *, end: str = "\n", style: str | None = None) -> None:
        self.console.print(Text(visible(text), style=style or ""), end=end, soft_wrap=True)

    async def read(
        self,
        label: str,
        *,
        commands: list[str] | None = None,
        toolbar: Callable[[], str] | None = None,
    ) -> str:
        if not self.interactive:
            line: str = await asyncio.to_thread(sys.stdin.readline)
            if not line:
                raise EOFError
            return line.rstrip("\r\n")
        if self._prompt is None:
            self._prompt = PromptSession()
        completer = WordCompleter(commands, sentence=True) if commands else None
        with patch_stdout():
            return await self._prompt.prompt_async(label, completer=completer, bottom_toolbar=toolbar)

    async def choose(self, label: str, choices: list[str]) -> int | None:
        if not choices or not self.interactive:
            return None
        async with self._menu_lock:
            for index, choice in enumerate(choices, 1):
                self.write(f"  {index}. {choice}")
            try:
                answer = (await self.read(f"{label} (number, Enter to cancel) > ")).strip()
            except (EOFError, KeyboardInterrupt):
                return None
            if answer.isdecimal() and len(answer) <= len(str(len(choices))):
                index = int(answer)
                if 1 <= index <= len(choices):
                    return index - 1
            return None

    def plan(self, entries: list[schema.PlanEntry]) -> None:
        self.write("Plan", style="bold")
        if not entries:
            self.write("No plan items.")
            return
        table = Table(show_header=False, box=None)
        table.add_column()
        table.add_column()
        for entry in entries:
            style = "green" if entry.status == "completed" else "yellow"
            table.add_row(Text(entry.status.replace("_", " "), style=style), Text(visible(entry.content)))
        self.console.print(table)

    def diff(self, content: schema.FileEditToolCallContent) -> None:
        diff = "".join(
            difflib.unified_diff(
                (content.old_text or "").splitlines(keepends=True),
                content.new_text.splitlines(keepends=True),
                fromfile=content.path,
                tofile=content.path,
            )
        )
        if diff:
            self.console.print(Syntax(visible(diff), "diff", theme="ansi_dark"))
