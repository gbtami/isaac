"""Rich output and prompt-toolkit menus for the terminal client."""

import asyncio
import difflib
import sys
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from acp import schema
from prompt_toolkit import PromptSession
from prompt_toolkit.application import get_app, in_terminal
from prompt_toolkit.completion import WordCompleter
from rich.console import Console, RenderableType
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
        self._line = Text()
        self._output: list[RenderableType] = []
        self._output_task: asyncio.Task[None] | None = None

    def write(self, text: str, *, end: str = "\n", style: str | None = None) -> None:
        text = visible(text)
        if not self.interactive:
            self.console.print(Text(text, style=style or ""), end=end, soft_wrap=True)
            return
        # Keep the unfinished line inside the prompt layout. The transcript
        # always leaves the cursor at the start of a new line for a redraw.
        complete, newline, tail = (text + end).rpartition("\n")
        if newline:
            self._line.append(complete, style)
            self._print(self._line)
            self._line = Text()
        self._line.append(tail, style)
        if self._prompt is not None and self._output_task is None:
            self._prompt.app.invalidate()

    def _print(self, renderable: RenderableType) -> None:
        self._output.append(renderable)
        if self._output_task is None:
            if self._prompt is not None and self._prompt.app.is_running:
                self._output_task = asyncio.create_task(self._print_above_prompt())
            else:
                self._print_output()

    def _print_output(self) -> None:
        with self.console:
            for renderable in self._output:
                self.console.print(renderable, soft_wrap=isinstance(renderable, Text))
        self._output.clear()

    async def _print_above_prompt(self) -> None:
        try:
            async with in_terminal():
                self._print_output()
        finally:
            self._output_task = None

    async def flush(self) -> None:
        if self._output_task is not None:
            await asyncio.shield(self._output_task)

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
        await self.flush()
        if self._prompt is None:
            self._prompt = PromptSession()
        self._prompt.app.erase_when_done = False
        completer = WordCompleter(commands, sentence=True) if commands else None

        def message() -> str:
            app = get_app()
            if self._line and not app.is_done:
                preview = self._line.plain.expandtabs(self.console.tab_size)
                limit = 2 * app.output.get_size().columns
                if len(preview) > limit:
                    preview = "…" + preview[1 - limit :]
                return f"{preview}\n{label}"
            return label

        def status_fallback() -> str:
            app = get_app()
            if toolbar and not app.renderer.height_is_known:
                width = max(0, app.output.get_size().columns - len(label) - 4)
                return toolbar()[:width]
            return ""

        return await self._prompt.prompt_async(
            message,
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
                self._prompt.app.erase_when_done = True
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
        self._print(table)

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
            self._print(Syntax(visible(diff), "diff", theme="ansi_dark"))
