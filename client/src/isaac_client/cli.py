"""Launch an independent ACP process and open a new or remembered session."""

import argparse
import asyncio
import os
import sqlite3
import sys
from contextlib import AsyncExitStack
from importlib.metadata import version
from pathlib import Path
from typing import cast

import acp
from acp import schema
from prompt_toolkit.application import create_app_session
from pydantic import TypeAdapter

from isaac_client.client import SessionState, TerminalClient
from isaac_client.display import Terminal
from isaac_client.history import SessionEntry, SessionHistory
from isaac_client.repl import interactive_loop

McpServer = schema.McpServerStdio | schema.HttpMcpServer | schema.SseMcpServer | schema.AcpMcpServer
MCP_SERVERS = TypeAdapter(list[McpServer])


async def run_client(
    command: list[str],
    cwd: Path,
    mcp_servers: list[McpServer],
    *,
    resume: bool = False,
    continue_latest: bool = False,
    session_id: str | None = None,
) -> int:
    terminal = Terminal()
    client = TerminalClient(terminal)
    async with AsyncExitStack() as stack:
        if terminal.interactive:
            # Share this app session with the ACP tasks that open approval and question menus.
            stack.enter_context(create_app_session())
            stack.push_async_callback(terminal.flush)
            stack.callback(client.finish_output)
        # The SDK routes unadvertised client operations independently of the required callbacks.
        connection, _process = await stack.enter_async_context(
            acp.spawn_agent_process(
                cast(acp.Client, client),
                *command,
                env=os.environ,
                transport_kwargs={"stderr": None},
                use_unstable_protocol=True,
            )
        )
        init = await connection.initialize(
            protocol_version=acp.PROTOCOL_VERSION,
            client_capabilities=schema.ClientCapabilities(
                session=schema.ClientSessionCapabilities(config_options=schema.SessionConfigOptionsCapabilities()),
                elicitation=schema.ElicitationCapabilities(form=schema.ElicitationFormCapabilities())
                if terminal.interactive
                else None,
            ),
            client_info=schema.Implementation(
                name="isaac-acp-client", title="Isaac Terminal Client", version=version("isaac-acp-client")
            ),
        )
        if init.protocol_version != acp.PROTOCOL_VERSION:
            raise RuntimeError(f"The agent requested ACP protocol version {init.protocol_version}.")
        if init.agent_info:
            client.agent = init.agent_info.title or init.agent_info.name
        capabilities = init.agent_capabilities or schema.AgentCapabilities()
        if capabilities.load_session:
            client.history = SessionHistory(command, cwd)
        if (resume or continue_latest or session_id is not None) and client.history is None:
            raise RuntimeError("This agent does not advertise session/load.")

        async def saved_sessions() -> list[SessionEntry]:
            assert client.history is not None
            if capabilities.session_capabilities and capabilities.session_capabilities.list is not None:
                try:
                    return await client.history.discover(connection)
                except acp.RequestError as exc:
                    terminal.write(f"Could not list agent sessions: {exc}", style="yellow")
            return client.history.recent()

        title = None
        if resume or continue_latest:
            entries = await saved_sessions()
            if not entries:
                raise RuntimeError("No sessions saved for this agent command and workspace.")
            if resume and not terminal.interactive:
                raise RuntimeError("The resume picker needs a terminal. Use --session ID for scripted input.")
            choice = (
                await terminal.choose("Resume session", [entry.label() for entry in entries], search=True)
                if resume
                else 0
            )
            if choice is None:
                return 0
            session_id = entries[choice].session_id
            title = entries[choice].title
        elif session_id is not None and client.history is not None:
            title = next((entry.title for entry in client.history.recent() if entry.session_id == session_id), None)

        async def load_session(target: str, target_title: str | None) -> schema.LoadSessionResponse:
            client.finish_output()
            previous = client.session
            client.session = SessionState(session_id=target, title=target_title, loading=True)
            try:
                return await connection.load_session(cwd=str(cwd), session_id=target, mcp_servers=mcp_servers)
            except BaseException:
                client.session = previous
                client.finish_output()
                raise

        def accept_session(response: schema.NewSessionResponse | schema.LoadSessionResponse) -> None:
            if isinstance(response, schema.NewSessionResponse):
                client.session.session_id = response.session_id
            if response.config_options is not None:
                client.session.options = response.config_options
            if response.modes is not None:
                client.session.modes = response.modes
            client.session.loading = False
            client.remember()
            client.finish_output()

        async def open_session() -> schema.NewSessionResponse | schema.LoadSessionResponse:
            if session_id is None:
                return await connection.new_session(cwd=str(cwd), mcp_servers=mcp_servers)
            return await load_session(session_id, title)

        async def resume_session() -> None:
            entries = [entry for entry in await saved_sessions() if entry.session_id != client.session.session_id]
            if not entries:
                terminal.write("No other saved sessions for this agent and workspace.")
                return
            choice = await terminal.choose("Resume session", [entry.label() for entry in entries], search=True)
            if choice is not None:
                entry = entries[choice]
                accept_session(await load_session(entry.session_id, entry.title))
                terminal.write(f"Session {entry.session_id}", style="dim")

        try:
            session = await open_session()
        except acp.RequestError as exc:
            methods = [method for method in init.auth_methods or [] if isinstance(method, schema.AuthMethodAgent)]
            if exc.code != -32000 or not methods:
                raise
            choice = await terminal.choose("Authenticate with", [method.name for method in methods])
            if choice is None:
                raise
            await connection.authenticate(method_id=methods[choice].id)
            session = await open_session()

        accept_session(session)
        terminal.write(f"{client.agent} · {cwd}")
        hint = "Type / for commands" if terminal.interactive else "/client:help"
        terminal.write(f"Session {client.session.session_id} · {hint}", style="dim")
        await interactive_loop(connection, client, resume_session)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Connect to an ACP agent over stdio.")
    parser.add_argument("--project-dir", type=Path, default=Path.cwd(), help="Workspace sent to the ACP agent")
    parser.add_argument("--mcp-config", type=Path, help="JSON file with an array of ACP MCP server definitions")
    session = parser.add_mutually_exclusive_group()
    session.add_argument("--resume", action="store_true", help="Choose a saved session for this agent and workspace")
    session.add_argument("--continue", dest="continue_latest", action="store_true", help="Open the most recent session")
    session.add_argument("--session", help="Open an ACP session by ID")
    parser.add_argument("agent", help="Executable or command to launch")
    parser.add_argument("agent_args", nargs=argparse.REMAINDER, help="Arguments passed to the agent")
    args = parser.parse_args()
    cwd = args.project_dir.expanduser().resolve()
    if not cwd.is_dir():
        parser.error(f"workspace directory does not exist: {cwd}")
    try:
        mcp_servers = MCP_SERVERS.validate_json(args.mcp_config.read_bytes()) if args.mcp_config else []
        return asyncio.run(
            run_client(
                [args.agent, *args.agent_args],
                cwd,
                mcp_servers,
                resume=args.resume,
                continue_latest=args.continue_latest,
                session_id=args.session,
            )
        )
    except KeyboardInterrupt:
        return 130
    except (acp.RequestError, OSError, RuntimeError, ValueError, sqlite3.Error) as exc:
        print(f"isaac-client: {exc}", file=sys.stderr)
        return 1
