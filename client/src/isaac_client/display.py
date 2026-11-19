"""Rich output and prompt-toolkit menus for the terminal client."""

import asyncio
import difflib
import sys
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from acp import schema
from prompt_toolkit import PromptSession
from prompt_toolkit.application import get_app
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text


class InputInterrupted(Exception):
    """The user pressed Ctrl+C in the prompt editor."""


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
        self._pending_menus = 0
        self._editor: asyncio.Task[str] | None = None
        self._draft = ""

    def write(self, text: str, *, end: str = "\n", style: str | None = None) -> None:
        self.console.print(Text(visible(text), style=style or ""), end=end, soft_wrap=True)

    async def read(
        self,
        label: str,
        *,
        commands: list[str] | None = None,
        toolbar: Callable[[], str] | None = None,
        default: str = "",
    ) -> str:
        if not self.interactive:
            line: str = await asyncio.to_thread(sys.stdin.readline)
            if not line:
                raise EOFError
            return line.rstrip("\r\n")
        if self._prompt is None:
            self._prompt = PromptSession()
        completer = WordCompleter(commands, sentence=True) if commands else None

        def status_fallback() -> str:
            app = get_app()
            if toolbar and not app.renderer.height_is_known:
                width = max(0, app.output.get_size().columns - len(label) - 4)
                return toolbar()[:width]
            return ""

        with patch_stdout():
            return await self._prompt.prompt_async(
                label,
                completer=completer,
                bottom_toolbar=toolbar or "",
                rprompt=status_fallback,
                default=default,
                refresh_interval=0.25 if toolbar else 0,
            )

    async def edit(self, label: str, *, commands: list[str], toolbar: Callable[[], str]) -> str:
        if not self.interactive:
            return await self.read(label, commands=commands, toolbar=toolbar)

        async def read_draft() -> str:
            try:
                return await self.read(label, commands=commands, toolbar=toolbar, default=self._draft)
            except KeyboardInterrupt as exc:
                raise InputInterrupted from exc

        while True:
            async with self._menu_lock:
                if self._pending_menus:
                    continue
                editor = self._editor = asyncio.create_task(read_draft())
                try:
                    line = await asyncio.shield(editor)
                    self._draft = ""
                    return line
                except asyncio.CancelledError:
                    task = asyncio.current_task()
                    if task is not None and task.cancelling():
                        raise
                    # An approval or question temporarily took over the terminal.
                except (InputInterrupted, EOFError):
                    self._draft = ""
                    raise
                finally:
                    self._stop_editor()
                    await asyncio.gather(editor, return_exceptions=True)
                    self._editor = None

    def _stop_editor(self) -> None:
        if self._editor is not None and not self._editor.done():
            if self._prompt is not None and self._prompt.app.is_running:
                self._draft = self._prompt.default_buffer.text
            self._editor.cancel()

    @asynccontextmanager
    async def menu(self) -> AsyncIterator[None]:
        self._pending_menus += 1
        try:
            self._stop_editor()
            async with self._menu_lock:
                yield
        finally:
            self._pending_menus -= 1

    async def choose(
        self,
        label: str,
        choices: list[str],
        *,
        toolbar: Callable[[], str] | None = None,
        search: bool = False,
    ) -> int | None:
        if not choices or not self.interactive:
            return None
        async with self.menu():
            matches = list(enumerate(choices))
            hint = "number or search, Enter to cancel" if search else "number, Enter to cancel"
            while True:
                shown = matches[:20] if search else matches
                for index, (_, choice) in enumerate(shown, 1):
                    self.write(f"  {index}. {choice}")
                if not shown:
                    self.write("No matches. Try another search.")
                elif len(shown) < len(matches):
                    self.write(f"Showing {len(shown)} of {len(matches)}. Type a search to narrow the list.")
                try:
                    answer = (await self.read(f"{label} ({hint}) > ", toolbar=toolbar)).strip()
                except (EOFError, KeyboardInterrupt):
                    return None
                if answer.isdecimal() and len(answer) <= len(str(len(shown))):
                    index = int(answer)
                    if 1 <= index <= len(shown):
                        return shown[index - 1][0]
                if not search or not answer:
                    return None
                matches = [(i, choice) for i, choice in enumerate(choices) if answer.casefold() in choice.casefold()]

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
