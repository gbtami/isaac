"""Rich output and prompt-toolkit menus for the terminal client."""

import asyncio
import difflib
import sys
from collections.abc import AsyncIterator, Callable, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass

from acp import schema
from prompt_toolkit import PromptSession
from prompt_toolkit.application import get_app, get_app_or_none, in_terminal
from prompt_toolkit.completion import CompleteEvent, Completer, Completion, DummyCompleter
from prompt_toolkit.document import Document
from prompt_toolkit.enums import DEFAULT_BUFFER
from prompt_toolkit.filters import Condition, has_completions, has_focus, is_multiline
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from rich.console import Console, RenderableType
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from isaac_client.picker import pick


class InputInterrupted(Exception):
    """The user pressed Ctrl+C in the prompt editor."""


def visible(text: str) -> str:
    """Render untrusted ACP labels and chunks without terminal control sequences."""
    return "".join(
        character
        for character in text
        if character in "\t\n" or ord(character) >= 32 and not 127 <= ord(character) < 160
    )


def edit_diff(content: schema.FileEditToolCallContent) -> str:
    lines = difflib.unified_diff(
        (content.old_text or "").splitlines(keepends=True),
        content.new_text.splitlines(keepends=True),
        fromfile=content.path,
        tofile=content.path,
    )
    return "".join(line if line.endswith("\n") else f"{line}\n\\ No newline at end of file\n" for line in lines)


@dataclass(frozen=True)
class SlashCommand:
    name: str
    description: str
    hint: str = ""
    local: str | None = None


CommandSource = Callable[[], Iterable[SlashCommand]]


def slash_prefix(document: Document) -> str | None:
    text = document.text
    if document.is_cursor_at_the_end and text.startswith("/") and not any(char.isspace() for char in text):
        return text.casefold()
    return None


class SlashCompleter(Completer):
    def __init__(self, commands: CommandSource) -> None:
        self.commands = commands

    def get_completions(self, document: Document, complete_event: CompleteEvent) -> Iterable[Completion]:
        prefix = slash_prefix(document)
        if prefix is None:
            return
        for command in self.commands():
            if not command.name.casefold().startswith(prefix):
                continue
            detail = f"[{command.hint}]  {command.description}" if command.hint else command.description
            yield Completion(
                command.name + (" " if command.hint else ""),
                start_position=-len(document.text),
                display=command.name,
                display_meta=" ".join(visible(detail).split()),
            )


_prompt_keys = KeyBindings()


@_prompt_keys.add("escape", filter=has_completions)
def _dismiss_commands(event: KeyPressEvent) -> None:
    event.current_buffer.cancel_completion()


@_prompt_keys.add("escape", "enter", filter=has_focus(DEFAULT_BUFFER) & is_multiline)
def _insert_newline(event: KeyPressEvent) -> None:
    event.current_buffer.cancel_completion()
    event.current_buffer.newline()


@_prompt_keys.add("enter", filter=has_focus(DEFAULT_BUFFER) & (has_completions | is_multiline))
def _submit_draft(event: KeyPressEvent) -> None:
    buffer = event.current_buffer
    state = buffer.complete_state
    if state is None:
        buffer.validate_and_handle()
        return
    selected = state.current_completion
    if selected is None and any(item.text.rstrip() == buffer.text for item in state.completions):
        buffer.validate_and_handle()
        return
    completion = selected or state.completions[0]
    buffer.apply_completion(completion)
    if selected is not None and not completion.text.endswith(" "):
        buffer.validate_and_handle()


class Terminal:
    def __init__(self) -> None:
        self.console = Console(markup=False, highlight=False)
        self.interactive = sys.stdin.isatty()
        self._composer: PromptSession[str] | None = None
        self._menu_lock = asyncio.Lock()
        self._pending_menus = 0
        self._editor: asyncio.Task[str] | None = None
        self._draft = Document()
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
        self.refresh_prompt()

    def refresh_prompt(self) -> None:
        # A queued print redraws after writing above the prompt.
        app = get_app_or_none()
        if app is not None and self._output_task is None:
            app.invalidate()

    def _print(self, renderable: RenderableType) -> None:
        self._output.append(renderable)
        if self._output_task is None:
            app = get_app_or_none()
            if self.interactive and app is not None and app.is_running:
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
        commands: CommandSource | None = None,
        toolbar: Callable[[], str] | None = None,
        default: str | Document = "",
        multiline: bool = False,
    ) -> str:
        if not self.interactive:
            line: str = await asyncio.to_thread(sys.stdin.readline)
            if not line:
                raise EOFError
            return line.rstrip("\r\n")
        await self.flush()
        if multiline:
            if self._composer is None:
                self._composer = PromptSession()
            prompt = self._composer
            prompt.app.erase_when_done = False
        else:
            prompt = PromptSession()
        completer = SlashCompleter(commands) if commands else DummyCompleter()

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

        # ACP updates invalidate the composer. A refresh timer can clear a native selection.
        return await prompt.prompt_async(
            message,
            completer=completer,
            complete_while_typing=Condition(
                lambda: commands is not None and slash_prefix(get_app().current_buffer.document) is not None
            ),
            reserve_space_for_menu=6,
            key_bindings=_prompt_keys,
            multiline=multiline,
            bottom_toolbar=toolbar or "",
            rprompt=status_fallback,
            default=default,
        )

    async def edit(self, label: str, *, commands: CommandSource | None, toolbar: Callable[[], str]) -> str:
        if not self.interactive:
            return await self.read(label, commands=commands, toolbar=toolbar)

        async def read_draft() -> str:
            try:
                return await self.read(label, commands=commands, toolbar=toolbar, default=self._draft, multiline=True)
            except KeyboardInterrupt as exc:
                raise InputInterrupted from exc

        while True:
            async with self._menu_lock:
                if self._pending_menus:
                    continue
                editor = self._editor = asyncio.create_task(read_draft())
                try:
                    line = await asyncio.shield(editor)
                    self._draft = Document()
                    return line
                except asyncio.CancelledError:
                    task = asyncio.current_task()
                    if task is not None and task.cancelling():
                        raise
                    # An approval or question temporarily took over the terminal.
                except (InputInterrupted, EOFError):
                    self._draft = Document()
                    raise
                finally:
                    self._stop_editor()
                    await asyncio.gather(editor, return_exceptions=True)
                    self._editor = None

    def _stop_editor(self) -> None:
        if self._editor is not None and not self._editor.done():
            if self._composer is not None and self._composer.app.is_running:
                self._composer.default_buffer.cancel_completion()
                self._draft = self._composer.default_buffer.document
                self._composer.app.erase_when_done = True
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
            await self.flush()
            return await pick(
                " ".join(visible(label).split()),
                [" ".join(visible(choice).split()) for choice in choices],
                search=search,
                toolbar=toolbar,
            )

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
        if diff := edit_diff(content):
            self._print(Syntax(visible(diff), "diff", theme="ansi_dark"))
