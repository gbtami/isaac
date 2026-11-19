"""A keyboard picker for terminal menus."""

from collections.abc import Callable

from prompt_toolkit import PromptSession
from prompt_toolkit.application import get_app
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.keys import Keys
from rich.cells import cell_len, set_cell_size


async def pick(label: str, choices: list[str], *, search: bool, toolbar: Callable[[], str] | None = None) -> int | None:
    entries = list(enumerate(choices))
    selected: int | None = None
    keys = KeyBindings()

    def results(query: str) -> list[tuple[int, str]]:
        if not search or not query or query.isdecimal():
            return entries
        words = query.casefold().split()
        return [entry for entry in entries if all(word in entry[1].casefold() for word in words)]

    def selection(query: str, matches: list[tuple[int, str]]) -> int | None:
        if selected is not None:
            return selected
        if query.isdecimal() and len(query) <= len(str(len(matches))):
            number = int(query)
            if 1 <= number <= len(matches):
                return number - 1
        if search and query and not query.isdecimal() and matches:
            return 0
        return None

    @keys.add("up")
    @keys.add("down")
    def move(event: KeyPressEvent) -> None:
        nonlocal selected
        query = event.current_buffer.text.strip()
        matches = results(query)
        if matches:
            direction = 1 if event.key_sequence[-1].key == Keys.Down else -1
            current = selection(query, matches)
            start = (0 if direction < 0 else -1) if current is None else current
            selected = (start + direction) % len(matches)

    @keys.add("enter")
    def accept(event: KeyPressEvent) -> None:
        query = event.current_buffer.text.strip()
        matches = results(query)
        position = selection(query, matches)
        if position is not None:
            event.app.exit(result=matches[position][0])
        elif not query:
            event.app.exit(result=None)

    @keys.add("escape", eager=True)
    def close(event: KeyPressEvent) -> None:
        event.app.exit(result=None)

    def display() -> str:
        app = get_app()
        size = app.output.get_size()
        query = app.current_buffer.text.strip()
        matches = results(query)
        position = selection(query, matches)
        limit = max(1, min(8, size.rows - 4))
        start = max(0, (position if position is not None else 0) - limit + 1)
        rows = [label, "↑/↓ select · Enter confirm · Esc close"]
        for offset, (_, choice) in enumerate(matches[start : start + limit], start):
            marker = "›" if offset == position else " "
            rows.append(f"{marker} {offset + 1}. {choice}")
        if not matches:
            rows.append("No matches")
        elif len(matches) > limit:
            rows.append(f"{start + 1}–{min(start + limit, len(matches))} of {len(matches)}")
        rows.append("Search ❯ " if search else "Choice ❯ ")
        width = max(1, size.columns - 1)
        return "\n".join(row if cell_len(row) <= width else set_cell_size(row, width - 1) + "…" for row in rows)

    prompt = PromptSession[int | None](display, key_bindings=keys, erase_when_done=True, bottom_toolbar=toolbar or "")

    def reset_selection(buffer: Buffer) -> None:
        nonlocal selected
        selected = None

    prompt.default_buffer.on_text_changed += reset_selection
    try:
        return await prompt.prompt_async()
    except (EOFError, KeyboardInterrupt):
        return None
