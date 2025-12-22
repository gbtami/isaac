"""Reusable ACP agent core with pluggable dependencies."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol

from acp import (
    Agent,
    AuthenticateResponse,
    CreateTerminalRequest,
    CreateTerminalResponse,
    InitializeResponse,
    KillTerminalCommandRequest,
    KillTerminalCommandResponse,
    LoadSessionResponse,
    NewSessionResponse,
    PromptResponse,
    ReadTextFileRequest,
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
    WriteTextFileRequest,
    WriteTextFileResponse,
    PROTOCOL_VERSION,
)
from acp.agent.connection import AgentSideConnection
from acp.contrib.tool_calls import ToolCallTracker
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
    PermissionOption,
    PromptCapabilities,
    SessionCapabilities,
    SessionListCapabilities,
    SessionInfo,
    SessionNotification,
    ToolCall,
    UserMessageChunk,
)

logger = logging.getLogger("acp_server")

DEFAULT_COMMAND_TIMEOUT_S = 30.0
DEFAULT_TOOL_OUTPUT_LIMIT = 48 * 1024


class PromptStrategy(Protocol):
    async def init_session(self, session_id: str, toolsets: list[Any], system_prompt: str | None = None) -> None: ...

    async def set_session_model(
        self, session_id: str, model_id: str, toolsets: list[Any], system_prompt: str | None = None
    ) -> None: ...

    async def handle_prompt(
        self,
        session_id: str,
        prompt_text: str,
        cancel_event: asyncio.Event,
    ) -> PromptResponse: ...

    def model_id(self, session_id: str) -> str | None: ...


class SessionStore(Protocol):
    def persist_meta(self, session_id: str, cwd: Path, mcp_servers: list[Any], *, current_mode: str) -> None: ...

    def load_meta(self, session_id: str) -> dict[str, Any]: ...

    def persist_update(self, session_id: str, note: SessionNotification) -> None: ...

    def load_history(self, session_id: str) -> list[SessionNotification]: ...

    def persist_strategy_state(self, session_id: str, data: dict[str, Any]) -> None: ...

    def load_strategy_state(self, session_id: str) -> dict[str, Any]: ...


@dataclass
class NullSessionStore:
    def persist_meta(self, session_id: str, cwd: Path, mcp_servers: list[Any], *, current_mode: str) -> None:
        _ = (session_id, cwd, mcp_servers, current_mode)

    def load_meta(self, session_id: str) -> dict[str, Any]:
        _ = session_id
        return {}

    def persist_update(self, session_id: str, note: SessionNotification) -> None:
        _ = (session_id, note)

    def load_history(self, session_id: str) -> list[SessionNotification]:
        _ = session_id
        return []

    def persist_strategy_state(self, session_id: str, data: dict[str, Any]) -> None:
        _ = (session_id, data)

    def load_strategy_state(self, session_id: str) -> dict[str, Any]:
        _ = session_id
        return {}


@dataclass
class SlashCommands:
    available_commands: Callable[[], list[Any]]
    handle_command: Callable[[Any, str, str], Awaitable[SessionNotification | None] | SessionNotification | None]


@dataclass
class PlanSupport:
    parse_request: Callable[[str], Any | None]
    build_notification: Callable[[str, Any], SessionNotification]


@dataclass
class ModelRegistry:
    current_model_id: Callable[[], str]
    set_current_model: Callable[[str], None]
    list_models: Callable[[], dict[str, dict[str, Any]]]
    get_context_limit: Callable[[str], int | None] | None = None


@dataclass
class UsageFormatter:
    normalize_usage: Callable[[Any], Any]
    format_usage_summary: Callable[[Any, int | None, str], str]


@dataclass
class FileSystemBackend:
    read_text_file: Callable[[dict[str, Path], ReadTextFileRequest], Awaitable[ReadTextFileResponse]]
    write_text_file: Callable[[dict[str, Path], WriteTextFileRequest], Awaitable[WriteTextFileResponse]]


@dataclass
class TerminalBackend:
    create_terminal: Callable[
        [dict[str, Path], dict[str, Any], CreateTerminalRequest], Awaitable[CreateTerminalResponse]
    ]
    terminal_output: Callable[[dict[str, Any], TerminalOutputRequest], Awaitable[TerminalOutputResponse]]
    wait_for_terminal_exit: Callable[
        [dict[str, Any], WaitForTerminalExitRequest], Awaitable[WaitForTerminalExitResponse]
    ]
    kill_terminal: Callable[[dict[str, Any], KillTerminalCommandRequest], Awaitable[KillTerminalCommandResponse]]
    release_terminal: Callable[[dict[str, Any], ReleaseTerminalRequest], Awaitable[ReleaseTerminalResponse]]


@dataclass
class ToolIO:
    truncate_text: Callable[[str, int], tuple[str, bool]] | None = None
    truncate_tool_output: Callable[[dict[str, Any], int], tuple[dict[str, Any], bool]] | None = None
    await_with_cancel: Callable[[Any, asyncio.Event], Awaitable[Any | None]] | None = None


@dataclass
class ACPAgentConfig:
    agent_name: str
    agent_title: str
    agent_version: str
    capabilities_builder: Callable[[], AgentCapabilities] | None = None
    build_system_prompt: Callable[[Path], str | None] | None = None
    prompt_strategy: PromptStrategy | None = None
    prompt_strategy_factory: Callable[["ACPAgentCore"], PromptStrategy] | None = None
    build_mcp_toolsets: Callable[[list[Any]], list[Any]] | None = None
    slash_commands: SlashCommands | None = None
    plan_support: PlanSupport | None = None
    model_registry: ModelRegistry | None = None
    usage_formatter: UsageFormatter | None = None
    session_store: SessionStore | None = None
    file_system: FileSystemBackend | None = None
    terminal: TerminalBackend | None = None
    tool_runner: Callable[..., Awaitable[dict[str, Any]]] | None = None
    tool_io: ToolIO = field(default_factory=ToolIO)
    build_mode_state: Callable[[dict[str, str], str, str], Any] | None = None
    ext_method_handlers: dict[
        str, Callable[["ACPAgentCore", dict[str, Any]], Awaitable[dict[str, Any]] | dict[str, Any]]
    ] = field(default_factory=dict)
    ext_notification_handler: Callable[[str, dict[str, Any]], Awaitable[None] | None] | None = None
    model_build_error: type[Exception] | tuple[type[Exception], ...] | None = None
    tool_output_limit: int = DEFAULT_TOOL_OUTPUT_LIMIT
    terminal_output_limit: int = DEFAULT_TOOL_OUTPUT_LIMIT
    command_timeout_s: float = DEFAULT_COMMAND_TIMEOUT_S
    run_command_tool_name: str = "run_command"
    ext_method_names: list[str] | None = None
    extract_prompt_text: Callable[[list[ContentBlock]], str] | None = None
    coerce_user_text: Callable[[Any], str | None] | None = None


class ACPAgentCore(Agent):
    """ACP agent core with configurable dependencies."""

    def __init__(
        self,
        config: ACPAgentConfig,
        conn: AgentSideConnection | None = None,
    ) -> None:
        self._config = config
        self._conn: AgentSideConnection | None = conn
        self._sessions: set[str] = set()
        self._session_cwds: dict[str, Path] = {}
        self._terminals: dict[str, Any] = {}
        self._session_modes: dict[str, str] = {}
        self._cancel_events: dict[str, asyncio.Event] = {}
        self._session_model_ids: dict[str, str] = {}
        self._session_history: dict[str, list[SessionNotification]] = {}
        self._session_allowed_commands: dict[str, set[tuple[str, str]]] = {}
        self._session_mcp_servers: dict[str, Any] = {}
        self._session_usage: dict[str, Any] = {}
        self._session_toolsets: dict[str, list[Any]] = {}
        self._client_capabilities: Any | None = None
        self._client_info: Any | None = None
        self._session_commands_advertised: set[str] = set()
        self._command_timeout_s = config.command_timeout_s
        self._tool_output_limit = config.tool_output_limit
        self._terminal_output_limit = config.terminal_output_limit
        self._session_store: SessionStore = config.session_store or NullSessionStore()
        self._session_last_chunk: dict[str, str | None] = {}
        self._session_system_prompts: dict[str, str | None] = {}
        self._prompt_strategy = self._build_prompt_strategy()

    def on_connect(self, conn: AgentSideConnection) -> None:  # type: ignore[override]
        """Capture connection when wiring via run_agent/connect_to_agent."""
        self._conn = conn

    def _build_prompt_strategy(self) -> PromptStrategy:
        if self._config.prompt_strategy is not None:
            return self._config.prompt_strategy
        if self._config.prompt_strategy_factory is not None:
            return self._config.prompt_strategy_factory(self)
        raise RuntimeError("prompt_strategy or prompt_strategy_factory required")

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: Any | None = None,
        client_info: Any | None = None,
        **_: Any,
    ) -> InitializeResponse:
        """Handle ACP initialize handshake (Initialization section)."""
        logger.info("Received initialize request: %s", protocol_version)
        if protocol_version != PROTOCOL_VERSION:
            logger.warning(
                "Protocol version mismatch requested=%s supported=%s",
                protocol_version,
                PROTOCOL_VERSION,
            )
        self._client_capabilities = client_capabilities
        self._client_info = client_info

        capabilities = (
            self._config.capabilities_builder()
            if self._config.capabilities_builder is not None
            else AgentCapabilities(
                load_session=True,
                prompt_capabilities=PromptCapabilities(
                    embedded_context=True,
                    image=False,
                    audio=False,
                ),
                session_capabilities=SessionCapabilities(list=SessionListCapabilities()),
            )
        )

        ext_method_names = self._ext_method_names()
        if ext_method_names:
            meta = getattr(capabilities, "field_meta", None) or {}
            existing = set(meta.get("extMethods", []) or [])
            meta["extMethods"] = sorted(existing | set(ext_method_names))
            capabilities.field_meta = meta

        return InitializeResponse(
            protocol_version=PROTOCOL_VERSION,
            agent_capabilities=capabilities,
            agent_info=self._agent_info(),
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
        session_system_prompt = self._build_session_system_prompt(cwd_path)
        self._session_system_prompts[session_id] = session_system_prompt
        self._cancel_events[session_id] = asyncio.Event()
        toolsets = self._build_mcp_toolsets(mcp_servers)
        self._session_toolsets[session_id] = toolsets
        await self._prompt_strategy.init_session(session_id, toolsets, system_prompt=session_system_prompt)
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
        mode_state = None
        if self._config.build_mode_state is not None:
            mode_state = self._config.build_mode_state(self._session_modes, session_id, current_mode="ask")
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
        session_system_prompt = self._build_session_system_prompt(cwd_path)
        self._session_system_prompts[session_id] = session_system_prompt
        self._cancel_events.setdefault(session_id, asyncio.Event())
        mcp_servers = mcp_servers or stored_meta.get("mcpServers", [])
        toolsets = self._build_mcp_toolsets(mcp_servers)
        self._session_toolsets[session_id] = toolsets
        await self._prompt_strategy.init_session(session_id, toolsets, system_prompt=session_system_prompt)
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
        previous_model_id = self._session_model_ids.get(session_id, self._current_model_id())
        try:
            await self._prompt_strategy.set_session_model(
                session_id,
                model_id,
                toolsets=self._session_toolsets.get(session_id, []),
                system_prompt=self._session_system_prompts.get(session_id),
            )
            self._session_model_ids[session_id] = model_id
            if self._config.model_registry is not None:
                with contextlib.suppress(Exception):
                    self._config.model_registry.set_current_model(model_id)
        except Exception as exc:  # noqa: BLE001
            if self._config.model_registry is not None:
                with contextlib.suppress(Exception):
                    self._config.model_registry.set_current_model(previous_model_id)
            if self._config.model_build_error is not None and isinstance(exc, self._config.model_build_error):
                raise
            if self._config.model_build_error is None:
                raise
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
        self._session_last_chunk[session_id] = None

        for block in prompt:
            tool_call = getattr(block, "tool_call", None)
            if tool_call:
                await self._handle_tool_call(session_id, tool_call)
                return PromptResponse(stop_reason="end_turn")

        prompt_text = self._extract_prompt_text(prompt)
        self._store_user_prompt(session_id, prompt)

        if self._config.slash_commands is not None:
            slash = self._config.slash_commands.handle_command(self, session_id, prompt_text)
            if asyncio.iscoroutine(slash):
                slash = await slash
            if slash:
                await self._send_update(slash)
                return PromptResponse(stop_reason="end_turn")

        if self._config.plan_support is not None:
            plan_request = self._config.plan_support.parse_request(prompt_text)
            if plan_request:
                await self._send_update(self._config.plan_support.build_notification(session_id, plan_request))
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

        if function_name == self._config.run_command_tool_name:
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
        if self._config.tool_runner is None:
            result = {"content": None, "error": f"No tool runner configured for {tool_name}"}
        else:
            runner = self._config.tool_runner
            await_with_cancel = self._config.tool_io.await_with_cancel
            if await_with_cancel is None:
                result = await runner(
                    tool_name, cwd=str(self._session_cwds.get(session_id, Path.cwd())), **(arguments or {})
                )
            else:
                result = await await_with_cancel(
                    runner(tool_name, cwd=str(self._session_cwds.get(session_id, Path.cwd())), **(arguments or {})),
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

        result_with_tool = dict(result)
        result_with_tool.setdefault("tool", tool_name)
        truncated = False
        truncator = self._config.tool_io.truncate_tool_output
        if truncator is not None:
            result_with_tool, truncated = truncator(result_with_tool, self._tool_output_limit)
            if truncated:
                result_with_tool["truncated"] = True
        status = "completed" if not result_with_tool.get("error") else "failed"
        summary = result_with_tool.get("content") or result_with_tool.get("error") or ""
        content_blocks: list[Any] = []
        if tool_name == "edit_file":
            new_text = result_with_tool.get("new_text")
            old_text = result_with_tool.get("old_text")
            path = arguments.get("path", "") if arguments else ""
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
            title=self._config.run_command_tool_name,
            status="in_progress",
            raw_input={"tool": self._config.run_command_tool_name, **arguments},
        )
        await self._send_update(session_notification(session_id, start))

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

        if self._config.terminal is None:
            progress = tracker.progress(
                external_id=tool_call_id,
                status="failed",
                raw_output={"content": None, "error": "Terminal backend not configured"},
                content=[tool_content(text_block("Terminal backend not configured"))],
            )
            await self._send_update(session_notification(session_id, progress))
            return

        try:
            create_resp = await self._config.terminal.create_terminal(
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

                out_resp = await self._config.terminal.terminal_output(
                    self._terminals,
                    TerminalOutputRequest(session_id=session_id, terminal_id=term_id),
                )
                chunk = out_resp.output or ""
                if chunk:
                    truncator = self._config.tool_io.truncate_text
                    if truncator is not None:
                        chunk, chunk_truncated = truncator(chunk, self._tool_output_limit)
                    else:
                        chunk_truncated = False
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
                await self._config.terminal.release_terminal(
                    self._terminals,
                    ReleaseTerminalRequest(session_id=session_id, terminal_id=term_id),
                )

        if not collected:
            with contextlib.suppress(Exception):
                final_out = await self._config.terminal.terminal_output(
                    self._terminals,
                    TerminalOutputRequest(session_id=session_id, terminal_id=term_id),
                )
                if final_out.output:
                    truncator = self._config.tool_io.truncate_text
                    if truncator is not None:
                        capped_output, chunk_truncated = truncator(final_out.output, self._tool_output_limit)
                    else:
                        capped_output, chunk_truncated = final_out.output, False
                    collected.append(capped_output)
                    truncated = truncated or final_out.truncated or chunk_truncated
                    exit_code = exit_code or (final_out.exit_status.exit_code if final_out.exit_status else None)

        full_output = "".join(collected).rstrip("\n")
        truncator = self._config.tool_io.truncate_text
        if truncator is not None:
            full_output, capped = truncator(full_output, self._tool_output_limit)
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
        if self._config.file_system is None:
            return ReadTextFileResponse(content="")
        params = ReadTextFileRequest(path=path, session_id=session_id, field_meta=kwargs or None)
        return await self._config.file_system.read_text_file(self._session_cwds, params)

    async def write_text_file(self, content: str, path: str, session_id: str, **kwargs: Any) -> WriteTextFileResponse:
        """Serve fs/write_text_file to clients (File System section)."""
        if self._config.file_system is None:
            return WriteTextFileResponse()
        params = WriteTextFileRequest(content=content, path=path, session_id=session_id, field_meta=kwargs or None)
        return await self._config.file_system.write_text_file(self._session_cwds, params)

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
        if self._config.terminal is None:
            return CreateTerminalResponse(terminal_id="")
        params = CreateTerminalRequest(
            command=command,
            session_id=session_id,
            args=args,
            cwd=cwd,
            env=env,
            output_byte_limit=output_byte_limit,
            field_meta=kwargs or None,
        )
        return await self._config.terminal.create_terminal(self._session_cwds, self._terminals, params)

    async def terminal_output(self, session_id: str, terminal_id: str, **kwargs: Any) -> TerminalOutputResponse:
        """Stream terminal output (Terminals section)."""
        if self._config.terminal is None:
            return TerminalOutputResponse(output="", truncated=False, exit_status=None)
        params = TerminalOutputRequest(session_id=session_id, terminal_id=terminal_id, field_meta=kwargs or None)
        return await self._config.terminal.terminal_output(self._terminals, params)

    async def wait_for_terminal_exit(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> WaitForTerminalExitResponse:
        """Wait for a terminal to exit (Terminals section)."""
        if self._config.terminal is None:
            return WaitForTerminalExitResponse(exit_code=None, signal=None)
        params = WaitForTerminalExitRequest(session_id=session_id, terminal_id=terminal_id, field_meta=kwargs or None)
        return await self._config.terminal.wait_for_terminal_exit(self._terminals, params)

    async def kill_terminal_command(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> KillTerminalCommandResponse:
        """Kill a running terminal command (Terminals section)."""
        if self._config.terminal is None:
            return KillTerminalCommandResponse()
        params = KillTerminalCommandRequest(session_id=session_id, terminal_id=terminal_id, field_meta=kwargs or None)
        return await self._config.terminal.kill_terminal(self._terminals, params)

    async def release_terminal(self, session_id: str, terminal_id: str, **kwargs: Any) -> ReleaseTerminalResponse:
        """Release resources for a terminal (Terminals section)."""
        if self._config.terminal is None:
            return ReleaseTerminalResponse()
        params = ReleaseTerminalRequest(session_id=session_id, terminal_id=terminal_id, field_meta=kwargs or None)
        return await self._config.terminal.release_terminal(self._terminals, params)

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
            text = self._coerce_user_text(block)
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

    async def ext_method(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Handle extension methods."""
        handler = self._config.ext_method_handlers.get(name)
        if handler is not None:
            result = handler(self, payload)
            if asyncio.iscoroutine(result):
                return await result
            return result
        if self._config.model_registry is not None:
            if name == "model/list":
                session_id = payload.get("session_id")
                current = self._session_model_ids.get(session_id, self._current_model_id())
                models = self._config.model_registry.list_models()
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
        """Handle extension notifications."""
        handler = self._config.ext_notification_handler
        if handler is None:
            logger.info("Received ext notification %s params_keys=%s", method, sorted(params.keys()))
            return
        result = handler(method, params)
        if asyncio.iscoroutine(result):
            await result

    def _persist_session_meta(
        self,
        session_id: str,
        cwd: Path,
        mcp_servers: list[Any] | None,
        *,
        current_mode: str,
    ) -> None:
        self._session_store.persist_meta(session_id, cwd, list(mcp_servers or []), current_mode=current_mode)

    def _load_session_meta(self, session_id: str) -> dict[str, Any]:
        return self._session_store.load_meta(session_id)

    def _load_session_history(self, session_id: str) -> list[SessionNotification]:
        return self._session_store.load_history(session_id)

    def _build_usage_note(self, session_id: str) -> SessionNotification:
        """Build a usage summary for the current session on demand."""
        formatter = self._config.usage_formatter
        model_registry = self._config.model_registry
        if formatter is None or model_registry is None:
            return session_notification(
                session_id,
                update_agent_message(text_block("Usage tracking not configured.")),
            )
        model_id = self._session_model_ids.get(session_id, "")
        context_limit = None
        if model_registry.get_context_limit is not None:
            context_limit = model_registry.get_context_limit(model_id)
        usage = formatter.normalize_usage(self._session_usage.get(session_id))
        summary = formatter.format_usage_summary(usage, context_limit, model_id)
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
        if self._config.slash_commands is None:
            return
        commands = self._config.slash_commands.available_commands()
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
                title=f"{self._config.run_command_tool_name}: {command_display}",
                kind="execute",
                raw_input={"tool": self._config.run_command_tool_name, "command": command, "cwd": cwd},
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

    def _build_session_system_prompt(self, cwd: Path) -> str | None:
        builder = self._config.build_system_prompt
        if builder is None:
            return None
        return builder(cwd)

    def _build_mcp_toolsets(self, mcp_servers: list[Any]) -> list[Any]:
        builder = self._config.build_mcp_toolsets
        if builder is None:
            return []
        return builder(mcp_servers)

    def _current_model_id(self) -> str:
        if self._config.model_registry is None:
            return ""
        return self._config.model_registry.current_model_id()

    def _set_usage(self, session_id: str, usage: Any | None) -> None:
        self._session_usage[session_id] = usage

    def _agent_info(self) -> Any:
        from acp.schema import Implementation

        return Implementation(
            name=self._config.agent_name,
            title=self._config.agent_title,
            version=self._config.agent_version,
        )

    def _ext_method_names(self) -> list[str]:
        if self._config.ext_method_names is not None:
            return list(self._config.ext_method_names)
        names = set(self._config.ext_method_handlers.keys())
        if self._config.model_registry is not None:
            names.update({"model/list", "model/set"})
        return sorted(names)

    def _extract_prompt_text(self, prompt: list[ContentBlock]) -> str:
        if self._config.extract_prompt_text is not None:
            return self._config.extract_prompt_text(prompt)
        chunks: list[str] = []
        for block in prompt:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                chunks.append(text)
        return "\n".join(chunks).strip()

    def _coerce_user_text(self, block: Any) -> str | None:
        if self._config.coerce_user_text is not None:
            return self._config.coerce_user_text(block)
        text = getattr(block, "text", None)
        if isinstance(text, str):
            return text
        return None
