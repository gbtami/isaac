"""Drive prompt turns and client-local slash commands."""

import asyncio
from contextlib import suppress

import acp
from acp.client.connection import ClientSideConnection

from isaac_client.client import TerminalClient

COMMANDS = {
    "/client:model": "Choose the model offered by the agent",
    "/client:config": "Change an advertised session setting",
    "/client:plan": "Show the latest agent plan",
    "/client:help": "List client and agent commands",
    "/client:quit": "Exit",
}


async def interactive_loop(connection: ClientSideConnection, session_id: str, client: TerminalClient) -> None:
    terminal = client.terminal
    while True:
        commands = [*COMMANDS, *(f"/{item.name}" for item in client.commands)]
        try:
            line = (await terminal.read("❯ ", commands=commands, toolbar=client.toolbar)).strip()
        except EOFError:
            return
        except KeyboardInterrupt:
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
                try:
                    response = await connection.prompt(session_id=session_id, prompt=[acp.text_block(line)])
                    if response.stop_reason != "end_turn":
                        client.finish_message()
                        terminal.write(f"Stopped: {response.stop_reason}", style="yellow")
                except asyncio.CancelledError:
                    with suppress(acp.RequestError, OSError, TimeoutError):
                        await asyncio.wait_for(connection.cancel(session_id=session_id), timeout=2)
                    raise
        except acp.RequestError as exc:
            client.finish_message()
            terminal.write(str(exc), style="red")
        finally:
            client.finish_message()
