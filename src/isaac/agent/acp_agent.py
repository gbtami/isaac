"""Core ACP-compliant agent implementation.

Contains session/prompt handling, tool execution, filesystem/terminal support,
and ACP lifecycle handlers. Non-protocol conveniences live in agent.py.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict

from acp import (
    Agent,
    AuthenticateResponse,
    CreateTerminalRequest,
    CreateTerminalResponse,
    InitializeResponse,
    LoadSessionResponse,
    NewSessionResponse,
    KillTerminalCommandRequest,
    KillTerminalCommandResponse,
    PROTOCOL_VERSION,
    PromptResponse,
    ReadTextFileResponse,
    ReleaseTerminalRequest,
    ReleaseTerminalResponse,
    RequestError,
    RequestPermissionResponse,
    SetSessionModeResponse,
    SetSessionModelResponse,
    TerminalOutputRequest,
    TerminalOutputResponse,
    WaitForTerminalExitRequest,
    WaitForTerminalExitResponse,
    WriteTextFileResponse,
)
from acp.agent.connection import AgentSideConnection
from acp.helpers import (
    ContentBlock,
    session_notification,
    text_block,
    tool_content,
    tool_diff_content,
    update_agent_message,
)
from acp.schema import (
    AgentCapabilities,
    AvailableCommandsUpdate,
    CurrentModeUpdate,
    ListSessionsResponse,
    ReadTextFileRequest,
    SessionCapabilities,
    SessionListCapabilities,
    WriteTextFileRequest,
    Implementation,
    McpCapabilities,
    PromptCapabilities,
    PermissionOption,
    SessionInfo,
    SessionNotification,
    ToolCall,
    UserMessageChunk,
)
from acp.contrib.tool_calls import ToolCallTracker
from isaac.agent import models as model_registry
from isaac.agent.agent_terminal import (
    TerminalState,
    create_terminal,
    kill_terminal,
    release_terminal,
    terminal_output,
    wait_for_terminal_exit,
)
from isaac.agent.brain.strategy_runner import StrategyEnv
from isaac.agent.constants import TOOL_OUTPUT_LIMIT
from isaac.agent.fs import read_text_file, write_text_file
from isaac.agent.mcp_support import build_mcp_toolsets
from isaac.agent.planner import build_plan_notification, parse_plan_request
from isaac.agent.prompt_utils import coerce_user_text, extract_prompt_text
from isaac.agent.session_modes import build_mode_state
from isaac.agent.session_store import SessionStore
from isaac.agent.slash import available_slash_commands, handle_slash_command
from isaac.agent.tool_io import await_with_cancel, truncate_text, truncate_tool_output
from isaac.agent.tools import run_tool
from isaac.agent.runner import register_tools
from isaac.agent.usage import format_usage_summary, normalize_usage
from isaac.agent.brain.strategy_base import ModelBuildError, PromptStrategy
from isaac.agent.brain.handoff_strategy import HandoffPromptStrategy
from isaac.agent.brain.subagent_strategy import SubagentPromptStrategy

logger = logging.getLogger("acp_server")


class ACPAgent(Agent):
    """Implements ACP session, prompt, tool, filesystem, and terminal flows."""

    def __init__(
        self,
        conn: AgentSideConnection | None = None,
        *,
        agent_name: str = "isaac",
        agent_title: str = "Isaac ACP Agent",
        agent_version: str = "0.1.0",
        prompt_strategy: PromptStrategy | None = None,
    ) -> None:
        self._conn: AgentSideConnection | None = conn
        self._sessions: set[str] = set()
        self._session_cwds: dict[str, Path] = {}
        self._terminals: Dict[str, TerminalState] = {}
        self._session_modes: Dict[str, str] = {}
        self._cancel_events: Dict[str, asyncio.Event] = {}
        self._session_model_ids: Dict[str, str] = {}
        self._session_history: Dict[str, list[SessionNotification]] = {}
        self._session_allowed_commands: Dict[str, set[tuple[str, str]]] = {}
        self._session_mcp_servers: Dict[str, Any] = {}
        self._session_usage: Dict[str, Any] = {}
        self._session_toolsets: Dict[str, list[Any]] = {}
        self._agent_name = agent_name
        self._agent_title = agent_title
        self._agent_version = agent_version
        self._client_capabilities: Any | None = None
        self._client_info: Any | None = None
        self._session_commands_advertised: set[str] = set()
        self._command_timeout_s = 30.0
        self._tool_output_limit = TOOL_OUTPUT_LIMIT
        self._terminal_output_limit = TOOL_OUTPUT_LIMIT
        self._session_store = SessionStore(Path.home() / ".isaac" / "sessions")
        self._session_last_chunk: Dict[str, str | None] = {}
        self._prompt_strategy = prompt_strategy or self._build_prompt_strategy()

    def on_connect(self, conn: AgentSideConnection) -> None:  # type: ignore[override]
        """Capture connection when wiring via run_agent/connect_to_agent."""
        self._conn = conn

    def _build_prompt_strategy(self) -> PromptStrategy:
        """Construct the default planning/execution strategy."""

        strategy_name = (os.getenv("ISAAC_PROMPT_STRATEGY") or "subagent").strip().lower()
        env = StrategyEnv(
            session_modes=self._session_modes,
            session_last_chunk=self._session_last_chunk,
            send_update=self._send_update,
            request_run_permission=lambda session_id, tool_call_id, command, cwd: self._request_run_permission(  # type: ignore[arg-type]
                session_id=session_id,
                tool_call_id=tool_call_id,
                command=command,
                cwd=cwd,
            ),
            set_usage=lambda session_id, usage: self._session_usage.__setitem__(session_id, usage),
        )
        if strategy_name == "subagent":
            return SubagentPromptStrategy(
                env=env,
                register_tools=register_tools,
            )
        return HandoffPromptStrategy(
            env=env,
            register_tools=register_tools,
        )

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: Any | None = None,
        client_info: Any | None = None,
        **_: Any,
    ) -> InitializeResponse:
        """Handle ACP initialize handshake (Initialization section)."""
        logger.info("Received initialize request: %s", protocol_version)
        # ACP version negotiation: if the requested version isn't supported, respond with
        # the latest version we support; the client may disconnect if it can't accept it.
        if protocol_version != PROTOCOL_VERSION:
            logger.warning(
                "Protocol version mismatch requested=%s supported=%s",
                protocol_version,
                PROTOCOL_VERSION,
            )
        # Capture peer capabilities/info for optional behavior gating.
        self._client_capabilities = client_capabilities
        self._client_info = client_info
        capabilities = AgentCapabilities(
            load_session=True,
            prompt_capabilities=PromptCapabilities(
                embedded_context=False,
                image=False,
                audio=False,
            ),
            mcp_capabilities=McpCapabilities(http=True, sse=True),
            session_capabilities=SessionCapabilities(list=SessionListCapabilities()),
        )
        capabilities.field_meta = {"extMethods": ["model/list", "model/set"]}

        return InitializeResponse(
            protocol_version=PROTOCOL_VERSION,
            agent_capabilities=capabilities,
            agent_info=Implementation(
                name=self._agent_name,
                title=self._agent_title,
                version=self._agent_version,
            ),
        )

    async def authenticate(self, method_id: str, **_: Any) -> AuthenticateResponse | None:
        """Return a no-op authentication response (Initialization auth step)."""
        logger.info("Received authenticate request %s", method_id)
        return AuthenticateResponse()

    async def request_permission(
        self, options: list[PermissionOption], session_id: str, tool_call: ToolCall, **_: Any
    ) -> RequestPermissionResponse | None:
        """Return a permission outcome, following Prompt Turn guidance for gated tools."""
        if self._conn is None:
            raise RuntimeError("Connection not established")
        requester = getattr(self._conn, "request_permission", None)
        if requester is None:
            raise RuntimeError("Connection missing request_permission handler")
        return await requester(options=options, session_id=session_id, tool_call=tool_call)

    async def new_session(
        self,
        cwd: str,
        mcp_servers: list[Any],
        **_: Any,
    ) -> NewSessionResponse:
        """Create a new session (Session Setup / creation)."""
        session_id = str(uuid.uuid4())
        logger.info("Received new session request: %s cwd=%s", session_id, cwd)
        self._sessions.add(session_id)
        cwd_path = self._require_absolute_cwd(cwd)
        self._session_cwds[session_id] = cwd_path
        self._cancel_events[session_id] = asyncio.Event()
        toolsets = build_mcp_toolsets(mcp_servers)
        self._session_toolsets[session_id] = toolsets
        await self._prompt_strategy.init_session(session_id, toolsets)
        self._session_history[session_id] = []
        self._session_allowed_commands[session_id] = set()
        self._session_mcp_servers[session_id] = mcp_servers
        model_id = self._prompt_strategy.model_id(session_id) or self._current_model_id()
        self._session_model_ids[session_id] = model_id
        self._persist_session_meta(
            session_id,
            cwd_path,
            mcp_servers,
            current_mode="ask",
        )
        mode_state = build_mode_state(self._session_modes, session_id, current_mode="ask")
        await self._send_available_commands(session_id)
        return NewSessionResponse(session_id=session_id, modes=mode_state)

    async def load_session(
        self,
        cwd: str,
        mcp_servers: list[Any],
        session_id: str,
        **_: Any,
    ) -> LoadSessionResponse | None:
        """Reload an existing session and replay history (Session Setup / loading)."""
        logger.info("Received load session request %s", session_id)
        self._sessions.add(session_id)
        stored_meta = self._load_session_meta(session_id)
        cwd_path = self._require_absolute_cwd(cwd)
        self._session_cwds[session_id] = cwd_path
        self._cancel_events.setdefault(session_id, asyncio.Event())
        mcp_servers = mcp_servers or stored_meta.get("mcpServers", [])
        toolsets = build_mcp_toolsets(mcp_servers)
        self._session_toolsets[session_id] = toolsets
        await self._prompt_strategy.init_session(session_id, toolsets)
        self._session_history.setdefault(session_id, [])
        self._session_allowed_commands.setdefault(session_id, set())
        self._session_mcp_servers[session_id] = mcp_servers
        mode = stored_meta.get("mode") or self._session_modes.get(session_id, "ask")
        self._session_modes[session_id] = mode
        model_id = self._prompt_strategy.model_id(session_id) or self._current_model_id()
        self._session_model_ids[session_id] = model_id
        history = self._load_session_history(session_id)
        self._session_history[session_id].extend(history)
        self._persist_session_meta(
            session_id,
            cwd_path,
            mcp_servers,
            current_mode=mode,
        )
        for note in history:
            await self._conn.session_update(session_id=note.session_id, update=note.update)  # type: ignore[arg-type]
        await self._send_available_commands(session_id)
        return LoadSessionResponse()

    async def list_sessions(self, cursor: str | None = None, cwd: str | None = None, **_: Any) -> ListSessionsResponse:
        """Return known sessions; minimal implementation without paging."""
        sessions: list[SessionInfo] = []
        for session_id in self._sessions:
            sessions.append(
                SessionInfo(
                    session_id=session_id,
                    cwd=str(self._session_cwds.get(session_id, cwd or Path.cwd())),
                )
            )
        return ListSessionsResponse(sessions=sessions, next_cursor=None)

    async def set_session_mode(self, mode_id: str, session_id: str, **_: Any) -> SetSessionModeResponse | None:
        """Update the current session mode and broadcast (Session Modes)."""
        logger.info(
            "Received set session mode request %s -> %s",
            session_id,
            mode_id,
        )
        self._session_modes[session_id] = mode_id
        await self._send_update(
            session_notification(
                session_id,
                CurrentModeUpdate(session_update="current_mode_update", current_mode_id=mode_id),
            )
        )
        self._persist_session_meta(
            session_id,
            self._session_cwds.get(session_id, Path.cwd()),
            self._session_mcp_servers.get(session_id, []),
            current_mode=mode_id,
        )
        return SetSessionModeResponse()

    async def set_session_model(self, model_id: str, session_id: str, **_: Any) -> SetSessionModelResponse | None:
        """Switch the backing model for a session."""
        logger.info("Received set session model request %s -> %s", session_id, model_id)
        previous_model_id = self._session_model_ids.get(session_id, model_registry.current_model_id())
        try:
            await self._prompt_strategy.set_session_model(
                session_id,
                model_id,
                toolsets=self._session_toolsets.get(session_id, []),
            )
            self._session_model_ids[session_id] = model_id
            with contextlib.suppress(Exception):
                model_registry.set_current_model(model_id)
        except ModelBuildError:
            with contextlib.suppress(Exception):
                model_registry.set_current_model(previous_model_id)
            raise
        return SetSessionModelResponse()

    async def prompt(
        self,
        prompt: list[ContentBlock],
        session_id: str,
        **_: Any,
    ) -> PromptResponse:
        """Process a prompt turn per Prompt Turn lifecycle (session/prompt)."""
        logger.info("Received prompt request for session: %s", session_id)
        cancel_event = self._cancel_events.setdefault(session_id, asyncio.Event())
        cancel_event.clear()
        # Reset last text chunk tracking for this prompt turn.
        self._session_last_chunk[session_id] = None

        for block in prompt:
            tool_call = getattr(block, "tool_call", None)
            if tool_call:
                await self._handle_tool_call(session_id, tool_call)
                return PromptResponse(stop_reason="end_turn")

        prompt_text = extract_prompt_text(prompt)
        # Persist the user prompt after capturing history so the current turn is not duplicated.
        self._store_user_prompt(session_id, prompt)

        slash = await handle_slash_command(self, session_id, prompt_text)
        if slash:
            await self._send_update(slash)
            return PromptResponse(stop_reason="end_turn")

        plan_request = parse_plan_request(prompt_text)
        if plan_request:
            await self._send_update(build_plan_notification(session_id, plan_request))
            return PromptResponse(stop_reason="end_turn")

        if cancel_event.is_set():
            return PromptResponse(stop_reason="cancelled")

        return await self._prompt_strategy.handle_prompt(
            session_id,
            prompt_text,
            cancel_event,
        )

    async def cancel(self, session_id: str, **_: Any) -> None:
        """Stop in-flight prompt/tool work for a session (Prompt Turn cancellation)."""
        logger.info("Received cancel notification for session %s", session_id)
        event = self._cancel_events.get(session_id)
        if event:
            event.set()

    async def _handle_tool_call(self, session_id: str, tool_call: Any) -> None:
        """Dispatch a tool call coming from the model (Tool Calls section)."""
        function_name = getattr(tool_call, "function", "")
        arguments = getattr(tool_call, "arguments", {}) or {}
        tool_call_id = getattr(tool_call, "tool_call_id", str(uuid.uuid4()))

        if function_name == "run_command":
            await self._execute_run_command_with_terminal(
                session_id,
                tool_call_id=tool_call_id,
                arguments=arguments,
            )
            return

        await self._execute_tool(
            session_id,
            tool_name=function_name,
            tool_call_id=tool_call_id,
            arguments=arguments,
        )

    async def _execute_tool(
        self,
        session_id: str,
        *,
        tool_name: str,
        tool_call_id: str | None = None,
        arguments: dict[str, Any] | None = None,
    ) -> None:
        """Execute a regular tool and stream ACP tool_call_update notifications."""
        tool_call_id = tool_call_id or str(uuid.uuid4())
        tracker = ToolCallTracker(id_factory=lambda: tool_call_id)
        logger.info(
            "Tool call start %s session=%s args_keys=%s",
            tool_name,
            session_id,
            sorted(arguments or {}),
        )
        start = tracker.start(
            external_id=tool_call_id,
            title=tool_name,
            status="in_progress",
            raw_input={"tool": tool_name, **(arguments or {})},
        )
        await self._send_update(session_notification(session_id, start))

        cancel_event = self._cancel_events.setdefault(session_id, asyncio.Event())
        result: dict[str, Any] | None = await await_with_cancel(
            run_tool(
                tool_name,
                cwd=str(self._session_cwds.get(session_id, Path.cwd())),
                **(arguments or {}),
            ),
            cancel_event,
        )
        if result is None:
            progress = tracker.progress(
                external_id=tool_call_id,
                status="failed",
                raw_output={"content": None, "error": "cancelled"},
                content=[tool_content(text_block("Cancelled"))],
            )
            await self._send_update(session_notification(session_id, progress))
            return

        # Include tool name for downstream clients to interpret plan tool output.
        result_with_tool = dict(result)
        result_with_tool.setdefault("tool", tool_name)
        result_with_tool, was_truncated = truncate_tool_output(result_with_tool, self._tool_output_limit)
        if was_truncated:
            result_with_tool["truncated"] = True
        status = "completed" if not result_with_tool.get("error") else "failed"
        summary = result_with_tool.get("content") or result_with_tool.get("error") or ""
        content_blocks: list[Any] = []
        if tool_name == "edit_file":
            new_text = result_with_tool.get("new_text")
            old_text = result_with_tool.get("old_text")
            path = arguments.get("path", "")
            if isinstance(new_text, str):
                with contextlib.suppress(Exception):
                    content_blocks.append(tool_diff_content(path, new_text, old_text))
        if not content_blocks:
            content_blocks = [tool_content(text_block(summary))]
        logger.info(
            "Tool call done %s session=%s status=%s summary_preview=%s",
            tool_name,
            session_id,
            status,
            str(summary)[:160].replace("\n", "\\n"),
        )
        progress = tracker.progress(
            external_id=tool_call_id,
            status=status,
            raw_output=result_with_tool,
            content=content_blocks,
        )
        await self._send_update(session_notification(session_id, progress))

    async def _execute_run_command_with_terminal(
        self,
        session_id: str,
        *,
        tool_call_id: str,
        arguments: dict[str, Any],
    ) -> None:
        """Run the run_command tool using an ACP terminal for streamed output."""
        tracker = ToolCallTracker(id_factory=lambda: tool_call_id)
        command = arguments.get("command") or ""
        cwd_arg = arguments.get("cwd")

        start = tracker.start(
            external_id=tool_call_id,
            title="run_command",
            status="in_progress",
            raw_input={"tool": "run_command", **arguments},
        )
        await self._send_update(session_notification(session_id, start))

        # Request permission before executing shell commands (ask mode only).
        mode = self._session_modes.get(session_id, "ask")
        if mode == "ask":
            allowed = await self._request_run_permission(
                session_id,
                tool_call_id=tool_call_id,
                command=command,
                cwd=cwd_arg,
            )
            if not allowed:
                progress = tracker.progress(
                    external_id=tool_call_id,
                    status="failed",
                    raw_output={"content": None, "error": "permission denied"},
                    content=[tool_content(text_block("Command blocked: permission denied"))],
                )
                await self._send_update(session_notification(session_id, progress))
                return

        cancel_event = self._cancel_events.setdefault(session_id, asyncio.Event())
        cancel_event.clear()
        timeout_s = float(arguments.get("timeout") or self._command_timeout_s)

        try:
            create_resp = await create_terminal(
                self._session_cwds,
                self._terminals,
                CreateTerminalRequest(
                    session_id=session_id,
                    command="bash",
                    args=["-lc", command],
                    cwd=cwd_arg,
                    output_byte_limit=self._terminal_output_limit,
                ),
            )
        except Exception as exc:  # pragma: no cover - defensive
            progress = tracker.progress(
                external_id=tool_call_id,
                status="failed",
                raw_output={"content": None, "error": f"Failed to start command: {exc}"},
                content=[tool_content(text_block(f"Failed to start command: {exc}"))],
            )
            await self._send_update(session_notification(session_id, progress))
            return

        term_id = create_resp.terminal_id
        collected: list[str] = []
        truncated = False
        exit_code: int | None = None
        error_msg: str | None = None

        try:
            start_time = asyncio.get_event_loop().time()
            while True:
                if cancel_event.is_set():
                    error_msg = "cancelled"
                    await self.kill_terminal_command(session_id=session_id, terminal_id=term_id)
                    break

                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed > timeout_s:
                    error_msg = f"Command timed out after {timeout_s}s"
                    await self.kill_terminal_command(session_id=session_id, terminal_id=term_id)
                    break

                out_resp = await terminal_output(
                    self._terminals,
                    TerminalOutputRequest(session_id=session_id, terminal_id=term_id),
                )
                chunk = out_resp.output or ""
                if chunk:
                    chunk, chunk_truncated = truncate_text(chunk, self._tool_output_limit)
                    collected.append(chunk)
                    truncated = truncated or out_resp.truncated or chunk_truncated
                    progress = tracker.progress(
                        external_id=tool_call_id,
                        status="in_progress",
                        raw_output={
                            "content": chunk,
                            "error": None,
                            "returncode": exit_code,
                            "truncated": out_resp.truncated or chunk_truncated,
                        },
                        content=[tool_content(text_block(chunk))],
                    )
                    await self._send_update(session_notification(session_id, progress))

                if out_resp.exit_status:
                    exit_code = out_resp.exit_status.exit_code
                    break

                await asyncio.sleep(0.2)
        finally:
            with contextlib.suppress(Exception):
                await release_terminal(
                    self._terminals,
                    ReleaseTerminalRequest(session_id=session_id, terminal_id=term_id),
                )

        if not collected:
            with contextlib.suppress(Exception):
                final_out = await terminal_output(
                    self._terminals,
                    TerminalOutputRequest(session_id=session_id, terminal_id=term_id),
                )
                if final_out.output:
                    capped_output, chunk_truncated = truncate_text(final_out.output, self._tool_output_limit)
                    collected.append(capped_output)
                    truncated = truncated or final_out.truncated or chunk_truncated
                    exit_code = exit_code or (final_out.exit_status.exit_code if final_out.exit_status else None)

        full_output, capped = truncate_text("".join(collected).rstrip("\n"), self._tool_output_limit)
        truncated = truncated or capped
        status = "failed" if error_msg else "completed"
        if exit_code not in (0, None) and not error_msg:
            status = "failed"

        summary = error_msg or full_output or ""
        progress = tracker.progress(
            external_id=tool_call_id,
            status=status,
            raw_output={
                "content": full_output,
                "error": error_msg,
                "returncode": exit_code,
                "truncated": truncated,
            },
            content=[tool_content(text_block(summary))] if summary else None,
        )
        await self._send_update(session_notification(session_id, progress))

    async def read_text_file(self, path: str, session_id: str, **kwargs: Any) -> ReadTextFileResponse:
        """Serve fs/read_text_file to clients (File System section)."""
        params = ReadTextFileRequest(path=path, session_id=session_id, field_meta=kwargs or None)
        return await read_text_file(self._session_cwds, params)

    async def write_text_file(self, content: str, path: str, session_id: str, **kwargs: Any) -> WriteTextFileResponse:
        """Serve fs/write_text_file to clients (File System section)."""
        params = WriteTextFileRequest(content=content, path=path, session_id=session_id, field_meta=kwargs or None)
        return await write_text_file(self._session_cwds, params)

    async def create_terminal(
        self,
        command: str,
        session_id: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: list[Any] | None = None,
        output_byte_limit: int | None = None,
        **kwargs: Any,
    ) -> CreateTerminalResponse:
        """Create a terminal on the agent host (Terminals section)."""
        params = CreateTerminalRequest(
            command=command,
            session_id=session_id,
            args=args,
            cwd=cwd,
            env=env,
            output_byte_limit=output_byte_limit,
            field_meta=kwargs or None,
        )
        return await create_terminal(self._session_cwds, self._terminals, params)

    async def terminal_output(self, session_id: str, terminal_id: str, **kwargs: Any) -> TerminalOutputResponse:
        """Stream terminal output (Terminals section)."""
        params = TerminalOutputRequest(session_id=session_id, terminal_id=terminal_id, field_meta=kwargs or None)
        return await terminal_output(self._terminals, params)

    async def wait_for_terminal_exit(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> WaitForTerminalExitResponse:
        """Wait for a terminal to exit (Terminals section)."""
        params = WaitForTerminalExitRequest(session_id=session_id, terminal_id=terminal_id, field_meta=kwargs or None)
        return await wait_for_terminal_exit(self._terminals, params)

    async def kill_terminal_command(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> KillTerminalCommandResponse:
        """Kill a running terminal command (Terminals section)."""
        params = KillTerminalCommandRequest(session_id=session_id, terminal_id=terminal_id, field_meta=kwargs or None)
        return await kill_terminal(self._terminals, params)

    async def release_terminal(self, session_id: str, terminal_id: str, **kwargs: Any) -> ReleaseTerminalResponse:
        """Release resources for a terminal (Terminals section)."""
        params = ReleaseTerminalRequest(session_id=session_id, terminal_id=terminal_id, field_meta=kwargs or None)
        return await release_terminal(self._terminals, params)

    async def _send_update(self, note: SessionNotification) -> None:
        """Record and emit a session/update notification for replay support."""
        self._record_update(note)
        if self._conn is None:
            raise RuntimeError("Connection not established")
        sender = getattr(self._conn, "session_update", None)
        if sender is None:
            raise RuntimeError("Connection missing session_update handler")
        await sender(session_id=note.session_id, update=note.update)

    def _record_update(self, note: SessionNotification) -> None:
        """Cache and persist updates for replay after restarts."""
        history = self._session_history.setdefault(note.session_id, [])
        history.append(note)
        self._session_store.persist_update(note.session_id, note)

    def _store_user_prompt(self, session_id: str, prompt_blocks: list[Any]) -> None:
        """Persist user prompt content for session/load replay."""
        for block in prompt_blocks:
            text = coerce_user_text(block)
            if text is None:
                continue
            try:
                chunk = UserMessageChunk(
                    session_update="user_message_chunk",
                    content=text_block(text),
                )
            except Exception:
                continue
            self._record_update(SessionNotification(session_id=session_id, update=chunk))

    async def checkpoint_session(self, session_id: str) -> SessionNotification:
        """Persist strategy state for later restore."""

        snapshot: dict[str, Any] = {}
        strategy_snapshot = getattr(self._prompt_strategy, "snapshot", None)
        if callable(strategy_snapshot):
            snapshot = strategy_snapshot(session_id)
        self._session_store.persist_strategy_state(session_id, snapshot)
        return session_notification(
            session_id,
            update_agent_message(text_block("Checkpoint saved.")),
        )

    async def restore_session_state(self, session_id: str) -> SessionNotification:
        """Restore strategy state from persisted snapshot."""

        snapshot = self._session_store.load_strategy_state(session_id)
        if not snapshot:
            return session_notification(
                session_id,
                update_agent_message(text_block("No checkpoint available.")),
            )
        restorer = getattr(self._prompt_strategy, "restore_snapshot", None)
        if callable(restorer):
            await restorer(session_id, snapshot)
        return session_notification(
            session_id,
            update_agent_message(text_block("Checkpoint restored.")),
        )

    def _current_model_id(self) -> str:
        return model_registry.current_model_id()

    async def ext_method(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Handle extension methods for model listing/selection."""
        if name == "model/list":
            session_id = payload.get("session_id")
            current = self._session_model_ids.get(session_id, self._current_model_id())
            models = model_registry.list_user_models()
            return {
                "current": current,
                "models": [{"id": mid, "description": meta.get("description", "")} for mid, meta in models.items()],
            }
        if name == "model/set":
            session_id = payload.get("session_id")
            model_id = payload.get("model_id")
            if not session_id or not model_id:
                return {"error": "session_id and model_id required"}
            try:
                await self.set_session_model(model_id, session_id)
                self._session_model_ids[session_id] = model_id
                return {"current": model_id}
            except Exception as exc:  # noqa: BLE001
                return {"error": str(exc)}
        return {"error": f"Unknown ext method: {name}"}

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        """Handle extension notifications (noop placeholder to satisfy ACP interface)."""
        logger.info("Received ext notification %s params_keys=%s", method, sorted(params.keys()))

    def _persist_session_meta(
        self,
        session_id: str,
        cwd: Path,
        mcp_servers: list[Any] | None,
        *,
        current_mode: str,
    ) -> None:
        self._session_store.persist_meta(session_id, cwd, mcp_servers or [], current_mode=current_mode)

    def _load_session_meta(self, session_id: str) -> dict[str, Any]:
        return self._session_store.load_meta(session_id)

    def _load_session_history(self, session_id: str) -> list[SessionNotification]:
        return self._session_store.load_history(session_id)

    def _build_usage_note(self, session_id: str) -> SessionNotification:
        """Build a usage summary for the current session on demand."""
        model_id = self._session_model_ids.get(session_id, "")
        context_limit = model_registry.get_context_limit(model_id)
        usage = normalize_usage(self._session_usage.get(session_id))
        summary = format_usage_summary(usage, context_limit, model_id)
        return session_notification(
            session_id,
            update_agent_message(text_block(summary)),
        )

    @staticmethod
    def _require_absolute_cwd(cwd: str) -> Path:
        path = Path(cwd or "").expanduser()
        if not path.is_absolute():
            raise RequestError.invalid_request({"message": "cwd must be an absolute path"})
        return path

    async def _send_available_commands(self, session_id: str) -> None:
        """Advertise slash commands using `available_commands_update` per ACP spec."""
        if session_id in self._session_commands_advertised:
            return
        commands = available_slash_commands()
        update = AvailableCommandsUpdate(session_update="available_commands_update", available_commands=commands)
        await self._send_update(session_notification(session_id, update))
        self._session_commands_advertised.add(session_id)

    async def _request_run_permission(
        self,
        session_id: str,
        *,
        tool_call_id: str,
        command: str,
        cwd: str | None,
    ) -> bool:
        """Ask the client for permission to run a shell command (ACP permission flow)."""

        key = (command.strip(), cwd or "")
        if key in self._session_allowed_commands.get(session_id, set()):
            return True

        try:
            options = [
                PermissionOption(option_id="allow_once", name="Allow once", kind="allow_once"),
                PermissionOption(option_id="allow_always", name="Allow this command", kind="allow_always"),
                PermissionOption(option_id="reject_once", name="Reject", kind="reject_once"),
            ]
            command_display = command.strip() or "<empty command>"
            cwd_display = cwd or str(self._session_cwds.get(session_id, Path.cwd()))
            tool_call = ToolCall(
                tool_call_id=tool_call_id,
                title=f"run_command: {command_display}",
                kind="execute",
                raw_input={"tool": "run_command", "command": command, "cwd": cwd},
                content=[
                    tool_content(
                        text_block(f"Command: {command_display}\nCWD: {cwd_display}"),
                    )
                ],
                status="pending",
            )
            from acp.schema import ToolCallUpdate  # type: ignore

            tool_call_update = ToolCallUpdate.model_validate(
                getattr(tool_call, "model_dump", lambda **_: tool_call)(by_alias=True)
            )
            requester = getattr(self._conn, "request_permission", None)
            if requester is None:
                raise RuntimeError("Connection missing request_permission handler")
            resp = await requester(options=options, session_id=session_id, tool_call=tool_call_update)
            outcome = getattr(resp, "outcome", None)
            option_id = ""
            if outcome is not None:
                option_id = getattr(outcome, "option_id", "")
            key = (command.strip(), cwd or "")
            if option_id == "allow_always":
                self._session_allowed_commands.setdefault(session_id, set()).add(key)
                return True
            if option_id == "allow_once":
                return True
            if key in self._session_allowed_commands.get(session_id, set()):
                return True
            return False
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("Permission request failed, denying by default: %s", exc)
            return False
