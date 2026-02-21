"""Interactive ACP client with REPL, tool calls, and terminal support."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import getpass
import logging
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable

import asyncio.subprocess as aio_subprocess
from acp import PROTOCOL_VERSION, RequestError
from acp.core import connect_to_agent
from acp.schema import AuthMethod, ClientCapabilities, FileSystemCapability, Implementation

from isaac.acp_runtime import ACP_STDIO_BUFFER_LIMIT_BYTES
from isaac.client.acp_client import ACPClient, apply_session_config_options
from isaac.client.auth import (
    auth_method_env_var_name,
    auth_method_link,
    auth_method_type,
    extract_auth_methods,
    extract_error_auth_methods,
    find_auth_method,
    select_auth_method,
)
from isaac.client.mcp_config import load_mcp_config
from isaac.client.repl import interactive_loop
from isaac.client.session_state import SessionUIState
from isaac.log_utils import build_log_config, configure_logging, log_event

logger = logging.getLogger(__name__)

MAX_AUTH_RESTARTS = 3


@dataclass(frozen=True)
class AuthRestartRequest:
    method_id: str
    env_var_name: str
    env_var_value: str


@dataclass(frozen=True)
class AuthAttemptResult:
    authenticated: bool
    restart: AuthRestartRequest | None = None


@dataclass
class _RuntimeConnection:
    proc: aio_subprocess.Process
    conn: Any
    auth_methods: list[AuthMethod | Any]


def _extract_agent_version(init_resp: Any) -> str | None:
    """Read agent version from initialize response across SDK model variations."""

    info = getattr(init_resp, "agent_info", None)
    if info is None:
        info = getattr(init_resp, "agentInfo", None)
    if info is None:
        return None
    version = getattr(info, "version", None)
    if version is None and isinstance(info, dict):
        version = info.get("version")
    if version is None:
        return None
    text = str(version).strip()
    return text or None


def _prompt_env_var_value(
    env_name: str,
    *,
    link: str | None,
    prompt_secret: Callable[[str], str] | None = None,
) -> str:
    if not sys.stdin.isatty():
        raise RuntimeError(f"{env_name} is required; restart this client with the variable set.")
    if link:
        print(f"Get a key here: {link}")
    secret_reader = prompt_secret or getpass.getpass
    value = secret_reader(f"Enter value for {env_name}: ").strip()
    if not value:
        raise RuntimeError(f"{env_name} was empty; cannot authenticate.")
    return value


async def _authenticate_if_needed(
    conn: Any,
    auth_methods: list[AuthMethod | Any],
    *,
    agent_env: dict[str, str],
    prompt_secret: Callable[[str], str] | None = None,
) -> AuthAttemptResult:
    if not auth_methods:
        return AuthAttemptResult(authenticated=False)
    method_id = select_auth_method(auth_methods)
    method = find_auth_method(auth_methods, method_id)
    method_kind = auth_method_type(method) if method is not None else "agent"
    if method_kind == "env_var" and method is not None:
        env_name = auth_method_env_var_name(method)
        if env_name and not agent_env.get(env_name):
            value = _prompt_env_var_value(
                env_name,
                link=auth_method_link(method),
                prompt_secret=prompt_secret,
            )
            log_event(logger, "client.authenticate.restart_required", method_id=method_id, env_var=env_name)
            return AuthAttemptResult(
                authenticated=False,
                restart=AuthRestartRequest(
                    method_id=method_id,
                    env_var_name=env_name,
                    env_var_value=value,
                ),
            )
    log_event(logger, "client.authenticate.request", method_id=method_id)
    await conn.authenticate(method_id=method_id)
    log_event(logger, "client.authenticate.success", method_id=method_id)
    return AuthAttemptResult(authenticated=True)


def _auth_methods_for_error(
    init_auth_methods: list[AuthMethod | Any],
    error_data: Any,
) -> list[AuthMethod | Any]:
    return extract_error_auth_methods(error_data) or init_auth_methods


async def _start_runtime_connection(
    *,
    spawn_program: str,
    spawn_args: list[str],
    spawn_env: dict[str, str],
    client_impl: ACPClient,
    state: SessionUIState,
) -> _RuntimeConnection:
    proc = await asyncio.create_subprocess_exec(
        spawn_program,
        *spawn_args,
        stdin=aio_subprocess.PIPE,
        stdout=aio_subprocess.PIPE,
        env=spawn_env,
        limit=ACP_STDIO_BUFFER_LIMIT_BYTES,
    )
    if proc.stdin is None or proc.stdout is None:
        raise RuntimeError("Agent process does not expose stdio pipes")
    conn = connect_to_agent(client_impl, proc.stdin, proc.stdout)
    init_resp = await conn.initialize(
        protocol_version=PROTOCOL_VERSION,
        client_capabilities=ClientCapabilities(
            fs=FileSystemCapability(read_text_file=False, write_text_file=False),
            terminal=True,
        ),
        client_info=Implementation(name="example-client", title="Example Client", version="0.3.1"),
    )
    if init_resp.protocol_version != PROTOCOL_VERSION:
        await _shutdown_runtime_connection(_RuntimeConnection(proc=proc, conn=conn, auth_methods=[]))
        raise RuntimeError(f"Incompatible ACP protocol version from agent: {init_resp.protocol_version}")
    state.agent_version = _extract_agent_version(init_resp)
    state.notify_changed()
    return _RuntimeConnection(proc=proc, conn=conn, auth_methods=extract_auth_methods(init_resp))


async def _shutdown_runtime_connection(runtime: _RuntimeConnection) -> None:
    with contextlib.suppress(Exception):
        await runtime.conn.close()
    if runtime.proc.returncode is None:
        runtime.proc.terminate()
        with contextlib.suppress(ProcessLookupError):
            await runtime.proc.wait()


async def run_client(program: str, args: Iterable[str], mcp_servers: list[Any]) -> int:
    _setup_client_logging()
    log_event(logger, "client.start", program=program)

    program_path = Path(program)
    spawn_program = program
    spawn_args = list(args)

    if program_path.exists() and not os.access(program_path, os.X_OK):
        spawn_program = sys.executable
        spawn_args = [str(program_path), *spawn_args]

    spawn_env = dict(os.environ)

    state = SessionUIState(current_mode="ask", current_model="unknown", mcp_servers=[])
    client_impl = ACPClient(state)
    runtime: _RuntimeConnection | None = None
    state.mcp_servers = [
        (srv.get("name") if isinstance(srv, dict) else getattr(srv, "name", "<server>")) or "<server>"
        for srv in mcp_servers
    ]
    cwd = os.getcwd()
    authenticated = False
    pending_auth_method: str | None = None
    restart_count = 0

    try:
        runtime = await _start_runtime_connection(
            spawn_program=spawn_program,
            spawn_args=spawn_args,
            spawn_env=spawn_env,
            client_impl=client_impl,
            state=state,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        while True:
            try:
                if pending_auth_method:
                    method_id = pending_auth_method
                    pending_auth_method = None
                    log_event(logger, "client.authenticate.request", method_id=method_id)
                    await runtime.conn.authenticate(method_id=method_id)
                    log_event(logger, "client.authenticate.success", method_id=method_id)
                    authenticated = True
                session = await runtime.conn.new_session(cwd=cwd, mcp_servers=mcp_servers)
                break
            except RequestError as exc:
                available_auth_methods = _auth_methods_for_error(runtime.auth_methods, getattr(exc, "data", None))
                if exc.code == -32000 and available_auth_methods and not authenticated:
                    auth_result = await _authenticate_if_needed(
                        runtime.conn,
                        available_auth_methods,
                        agent_env=spawn_env,
                    )
                    if auth_result.restart is not None:
                        restart_count += 1
                        if restart_count > MAX_AUTH_RESTARTS:
                            raise RuntimeError("Too many auth restarts; aborting.")
                        restart = auth_result.restart
                        spawn_env[restart.env_var_name] = restart.env_var_value
                        pending_auth_method = restart.method_id
                        await _shutdown_runtime_connection(runtime)
                        runtime = await _start_runtime_connection(
                            spawn_program=spawn_program,
                            spawn_args=spawn_args,
                            spawn_env=spawn_env,
                            client_impl=client_impl,
                            state=state,
                        )
                        continue
                    authenticated = auth_result.authenticated
                    continue
                raise
        state.session_id = session.session_id
        state.cwd = cwd
        if getattr(session, "config_options", None):
            apply_session_config_options(state, session.config_options or [])
        if getattr(session, "modes", None) and getattr(session.modes, "current_mode_id", None):
            state.current_mode = session.modes.current_mode_id
            state.notify_changed()
        if getattr(session, "models", None) and getattr(session.models, "current_model_id", None):
            state.current_model = session.models.current_model_id
            state.notify_changed()

        try:
            await interactive_loop(runtime.conn, session.session_id, state)
            return 0
        except KeyboardInterrupt:
            return 130
    finally:
        if runtime is not None:
            await _shutdown_runtime_connection(runtime)


async def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Run ACP client against an agent program.")
    parser.add_argument(
        "--mcp-config",
        type=str,
        help="Path to JSON file containing ACP mcpServers array (stdio/http/sse entries).",
    )
    parser.add_argument("agent_program", help="Path to the agent program to launch")
    parser.add_argument("agent_args", nargs=argparse.REMAINDER, help="Arguments for the agent")
    args = parser.parse_args(argv[1:])

    mcp_servers: list[Any] = []
    if args.mcp_config:
        mcp_servers = load_mcp_config(args.mcp_config)

    return await run_client(args.agent_program, args.agent_args, mcp_servers)


def _setup_client_logging() -> None:
    """Initialize client logging to a file (mirrors agent logging convention)."""

    config = build_log_config(log_file_name="acp_client.log")
    configure_logging(
        replace(
            config,
            logger_levels={
                "isaac": config.level,
                "acp": config.level,
            },
        )
    )
