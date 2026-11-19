"""Drive prompt turns and client-local slash commands."""

import asyncio
from collections import deque
from contextlib import suppress

import acp
from acp.client.connection import ClientSideConnection

from isaac_client.client import TerminalClient
from isaac_client.display import InputInterrupted

COMMANDS = {
    "/client:model": "Choose the model offered by the agent",
    "/client:config": "Change an advertised session setting",
    "/client:plan": "Show the latest agent plan",
    "/client:help": "List client and agent commands",
    "/client:quit": "Exit",
}


async def interactive_loop(connection: ClientSideConnection, session_id: str, client: TerminalClient) -> None:
    terminal = client.terminal
    queued: deque[str] = deque()
    while True:
        commands = [*COMMANDS, *(f"/{item.name}" for item in client.commands)]
        try:
            line = (
                queued.popleft()
                if queued
                else (await terminal.edit("❯ ", commands=commands, toolbar=client.toolbar)).strip()
            )
        except EOFError:
            return
        except InputInterrupted:
            continue
        if not line:
            continue

        command = line.split(maxsplit=1)[0]
        if command == "/client:quit":
            return

        try:
            if command in {"/client:config", "/client:model"}:
                await client.select_config(connection, session_id, "model" if command == "/client:model" else None)
            elif command == "/client:plan":
                terminal.plan(client.plan)
            elif command == "/client:help":
                for name, description in COMMANDS.items():
                    terminal.write(f"{name}  {description}")
                for item in client.commands:
                    terminal.write(f"/{item.name}  {item.description}")
            else:
                if client.title is None and not line.startswith("/"):
                    client.title = " ".join(line.split())[:80]
                    client.remember()
                await send_turn(connection, session_id, client, line, queued)
        except acp.RequestError as exc:
            client.finish_message()
            terminal.write(str(exc), style="red")
        finally:
            client.finish_message()


async def send_turn(
    connection: ClientSideConnection, session_id: str, client: TerminalClient, line: str, queued: deque[str]
) -> None:
    client.start_turn()
    editor = (
        asyncio.create_task(collect_followups(connection, session_id, client, queued))
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
    connection: ClientSideConnection, session_id: str, client: TerminalClient, queued: deque[str]
) -> None:
    def toolbar() -> str:
        return f"Ctrl+C stops  ·  Queued {len(queued)}  ·  {client.toolbar()}"

    while True:
        try:
            line = (await client.terminal.edit("Queue ❯ ", commands=[], toolbar=toolbar)).strip()
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
        if line:
            queued.append(line)
            client.finish_message()
            client.terminal.write(f"Queued follow-up {len(queued)}.", style="dim")
