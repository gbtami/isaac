"""Drive prompt turns and client-local slash commands."""

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
from contextlib import suppress

import acp
from acp.client.connection import ClientSideConnection

from isaac_client.client import TerminalClient
from isaac_client.display import CommandSource, InputInterrupted, SlashCommand, visible

COMMANDS = {
    "model": ("/pick-model", "Choose the model offered by the agent"),
    "config": ("/settings", "Change an advertised session setting"),
    "plan": (None, "Show the latest agent plan"),
    "resume": ("/sessions", "Resume another session in this workspace"),
    "help": (None, "List client and agent commands"),
    "quit": ("/exit", "Exit (Ctrl+D at an empty idle prompt)"),
}


def command_catalog(client: TerminalClient) -> dict[str, SlashCommand]:
    commands = {
        f"/{item.name}": SlashCommand(f"/{item.name}", item.description, item.input.root.hint if item.input else "")
        for item in client.session.commands
        if item.name and visible(item.name) == item.name and not any(char.isspace() for char in item.name)
    }
    for name, (alternate, description) in COMMANDS.items():
        if name == "resume" and (client.history is None or not client.terminal.interactive):
            continue
        shortcut = f"/{name}"
        alias = alternate if shortcut in commands else shortcut
        if alias is not None and alias not in commands:
            commands[alias] = SlashCommand(alias, description, local=name)
    return commands


async def interactive_loop(
    connection: ClientSideConnection, client: TerminalClient, resume_session: Callable[[], Awaitable[None]]
) -> None:
    terminal = client.terminal
    queued: deque[str] = deque()

    def commands() -> list[SlashCommand]:
        return list(command_catalog(client).values())

    while True:
        from_queue = bool(queued)
        try:
            line = (
                queued.popleft() if from_queue else await terminal.edit("❯ ", commands=commands, toolbar=client.toolbar)
            )
        except EOFError:
            return
        except InputInterrupted:
            continue
        if not terminal.interactive:
            line = line.strip()
        if not line.strip():
            continue

        entry = command_catalog(client).get(line.split(maxsplit=1)[0]) if "\n" not in line else None
        local = entry.local if entry else None
        if local == "quit":
            return

        session_id = client.session.session_id
        assert session_id is not None
        try:
            if local in {"config", "model"}:
                await client.select_config(connection, session_id, "model" if local == "model" else None)
            elif local == "plan":
                terminal.plan(client.session.plan)
            elif local == "resume":
                if from_queue:
                    terminal.write("Open the session picker from the idle prompt.", style="yellow")
                else:
                    await resume_session()
            elif local == "help":
                for item in commands():
                    hint = f" [{item.hint}]" if item.hint else ""
                    terminal.write(f"{item.name}{hint}  {item.description}")
            else:
                if client.session.title is None and not line.lstrip().startswith("/"):
                    client.session.title = " ".join(line.split())[:80]
                    client.remember()
                await send_turn(connection, session_id, client, line, queued, commands)
        except acp.RequestError as exc:
            client.finish_message()
            details = exc.data.get("details") if isinstance(exc.data, dict) else None
            terminal.write(f"{exc}: {details}" if isinstance(details, str) and details else str(exc), style="red")
        finally:
            client.finish_message()


async def send_turn(
    connection: ClientSideConnection,
    session_id: str,
    client: TerminalClient,
    line: str,
    queued: deque[str],
    commands: CommandSource,
) -> None:
    client.start_turn()
    editor = (
        asyncio.create_task(collect_followups(connection, session_id, client, queued, commands))
        if client.terminal.interactive
        else None
    )
    try:
        response = await connection.prompt(session_id=session_id, prompt=[acp.text_block(line)])
        if response.stop_reason != "end_turn":
            client.finish_message()
            client.terminal.write(f"Stopped: {response.stop_reason}", style="yellow")
    except asyncio.CancelledError:
        client.cancel_interactions()
        with suppress(acp.RequestError, OSError, TimeoutError):
            await asyncio.wait_for(connection.cancel(session_id=session_id), timeout=2)
        raise
    finally:
        if editor is not None:
            editor.cancel()
            await asyncio.gather(editor, return_exceptions=True)
        client.end_turn()
        client.cancelling = False


async def collect_followups(
    connection: ClientSideConnection,
    session_id: str,
    client: TerminalClient,
    queued: deque[str],
    commands: CommandSource,
) -> None:
    def toolbar() -> str:
        return f"Queued {len(queued)}  ·  Enter queues  ·  Ctrl+C stops  ·  {client.toolbar()}"

    while True:
        try:
            line = await client.terminal.edit("❯ ", commands=commands, toolbar=toolbar)
        except EOFError:
            return
        except InputInterrupted:
            client.cancel_interactions()
            queued.clear()
            client.finish_message()
            client.terminal.write("Cancelling turn…", style="yellow")
            try:
                await connection.cancel(session_id=session_id)
            except (acp.RequestError, OSError) as exc:
                client.terminal.write(str(exc), style="red")
            return
        if line.strip():
            queued.append(line)
            client.finish_message()
            client.terminal.write(f"Queued follow-up {len(queued)}.", style="dim")
