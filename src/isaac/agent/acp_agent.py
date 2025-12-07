"""Core ACP-compliant agent implementation.

Contains session/prompt handling, tool execution, filesystem/terminal support,
and ACP lifecycle handlers. Non-protocol conveniences live in agent.py.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
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
    RequestPermissionRequest,
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
    session_notification,
    text_block,
    tool_content,
    update_agent_message,
)
from acp.schema import (
    AgentCapabilities,
    AvailableCommandsUpdate,
    CurrentModeUpdate,
    ListSessionsResponse,
    ReadTextFileRequest,
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
from acp.helpers import plan_entry, update_plan
from pydantic_ai.messages import FunctionToolCallEvent, FunctionToolResultEvent, RetryPromptPart

from isaac.agent import models as model_registry
from isaac.agent.agent_terminal import (
    TerminalState,
    create_terminal,
    kill_terminal,
    release_terminal,
    terminal_output,
    wait_for_terminal_exit,
)
from isaac.agent.brain.history import build_chat_history
from isaac.agent.brain.planner import parse_plan_from_text
from isaac.agent.constants import TOOL_OUTPUT_LIMIT
from isaac.agent.default_runners import create_default_runners
from isaac.agent.fs import read_text_file, write_text_file
from isaac.agent.mcp_support import build_mcp_toolsets
from isaac.agent.planner import build_plan_notification, parse_plan_request
from isaac.agent.prompt_utils import coerce_user_text, extract_prompt_text, is_plan_only_prompt
from isaac.agent.runner import register_tools, run_with_runner, stream_with_runner
from isaac.agent.session_modes import build_mode_state
from isaac.agent.session_store import SessionStore
from isaac.agent.slash import available_slash_commands, handle_slash_command
from isaac.agent.tool_io import await_with_cancel, truncate_text, truncate_tool_output
from isaac.agent.tools import get_tools, run_tool
from isaac.agent.tools.run_command import (
    RunCommandContext,
    reset_run_command_context,
    set_run_command_context,
)
from isaac.agent.usage import format_usage_summary, normalize_usage

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
        ai_runner: Any | None = None,
        planning_runner: Any | None = None,
    ) -> None:
        self._conn: AgentSideConnection | None = conn
        self._sessions: set[str] = set()
        self._session_cwds: dict[str, Path] = {}
        self._terminals: Dict[str, TerminalState] = {}
        self._session_modes: Dict[str, str] = {}
        self._cancel_events: Dict[str, asyncio.Event] = {}
        self._session_models: Dict[str, Any | None] = {}
        self._session_planners: Dict[str, Any | None] = {}
        self._session_model_ids: Dict[str, str] = {}
        self._session_history: Dict[str, list[SessionNotification]] = {}
        self._session_allowed_commands: Dict[str, set[tuple[str, str]]] = {}
        self._session_mcp_servers: Dict[str, Any] = {}
        self._session_usage: Dict[str, Any] = {}
        self._session_toolsets: Dict[str, list[Any]] = {}
        self._agent_name = agent_name
        self._agent_title = agent_title
        self._agent_version = agent_version
        self._custom_runners = ai_runner is not None or planning_runner is not None
        self._session_commands_advertised: set[str] = set()
        self._ai_runner = ai_runner
        self._planning_runner = planning_runner
        self._command_timeout_s = 30.0
        self._tool_output_limit = TOOL_OUTPUT_LIMIT
        self._terminal_output_limit = TOOL_OUTPUT_LIMIT
        self._session_store = SessionStore(Path.home() / ".isaac" / "sessions")
        self._session_last_chunk: Dict[str, str | None] = {}
        self._session_model_errors: Dict[str, str] = {}
        self._session_model_error_notified: set[str] = set()

    def on_connect(self, conn: AgentSideConnection) -> None:  # type: ignore[override]
        """Capture connection when wiring via run_agent/connect_to_agent."""
        self._conn = conn

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: Any | None = None,
        client_info: Any | None = None,
        **_: Any,
    ) -> InitializeResponse:
        """Handle ACP initialize handshake (Initialization section)."""
        logger.info("Received initialize request: %s", protocol_version)
        capabilities = AgentCapabilities(
            load_session=True,
            prompt_capabilities=PromptCapabilities(
                embedded_context=True,
                image=False,
                audio=False,
            ),
            mcp_capabilities=McpCapabilities(http=True, sse=True),
        )
        capabilities.field_meta = {"extMethods": ["model/list", "model/set"]}
        try:
            capabilities.tools = get_tools()  # type: ignore[attr-defined]
        except Exception:
            pass

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
        # Support both snake_case (tests) and camelCase (ACP connection).
        requester = getattr(self._conn, "request_permission", None)
        if requester is None:
            requester = getattr(self._conn, "requestPermission", None)
        if requester is None:
            raise RuntimeError("Connection missing request_permission/requestPermission handler")
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
        cwd_path = Path(cwd or Path.cwd()).expanduser()
        if not cwd_path.is_absolute():
            cwd_path = (Path.cwd() / cwd_path).resolve()
        self._session_cwds[session_id] = cwd_path
        self._cancel_events[session_id] = asyncio.Event()
        toolsets = build_mcp_toolsets(mcp_servers)
        self._session_toolsets[session_id] = toolsets
        await self._init_session_runners(session_id, toolsets)
        self._session_history[session_id] = []
        self._session_allowed_commands[session_id] = set()
        self._session_mcp_servers[session_id] = mcp_servers
        self._persist_session_meta(session_id, cwd_path, mcp_servers, current_mode="ask")
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
        cwd_path = Path(cwd or stored_meta.get("cwd") or Path.cwd()).expanduser()
        if not cwd_path.is_absolute():
            cwd_path = (Path.cwd() / cwd_path).resolve()
        self._session_cwds[session_id] = cwd_path
        self._cancel_events.setdefault(session_id, asyncio.Event())
        mcp_servers = mcp_servers or stored_meta.get("mcpServers", [])
        toolsets = build_mcp_toolsets(mcp_servers)
        self._session_toolsets[session_id] = toolsets
        await self._init_session_runners(session_id, toolsets)
        self._session_history.setdefault(session_id, [])
        self._session_allowed_commands.setdefault(session_id, set())
        self._session_mcp_servers[session_id] = mcp_servers
        mode = stored_meta.get("mode") or self._session_modes.get(session_id, "ask")
        self._session_modes[session_id] = mode
        history = self._load_session_history(session_id)
        self._session_history[session_id].extend(history)
        self._persist_session_meta(session_id, cwd_path, mcp_servers, current_mode=mode)
        for note in history:
            await self._conn.session_update(session_id=note.session_id, update=note.update)  # type: ignore[arg-type]
        await self._send_available_commands(session_id)
        return LoadSessionResponse()

    async def list_sessions(
        self, cursor: str | None = None, cwd: str | None = None, **_: Any
    ) -> ListSessionsResponse:
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

    async def set_session_mode(
        self, mode_id: str, session_id: str, **_: Any
    ) -> SetSessionModeResponse | None:
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

    async def set_session_model(
        self, model_id: str, session_id: str, **_: Any
    ) -> SetSessionModelResponse | None:
        """Switch the backing model for a session."""
        logger.info("Received set session model request %s -> %s", session_id, model_id)
        try:
            toolsets = self._session_toolsets.get(session_id, [])
            executor, planner = model_registry.build_agent_pair(
                model_id,
                register_tools,
                toolsets=toolsets,
            )
            self._session_models[session_id] = executor
            self._session_planners[session_id] = planner
            self._session_model_ids[session_id] = model_id
            self._session_model_errors.pop(session_id, None)
            with contextlib.suppress(Exception):
                model_registry.set_current_model(model_id)
        except Exception as exc:  # pragma: no cover - model build errors
            logger.error("Failed to set session model: %s", exc)
            await self._send_update(
                session_notification(
                    session_id,
                    update_agent_message(text_block(f"Model load failed: {exc}")),
                )
            )
            raise
        return SetSessionModelResponse()

    async def prompt(
        self,
        prompt: list[Any],
        session_id: str,
        **_: Any,
    ) -> PromptResponse:
        """Process a prompt turn per Prompt Turn lifecycle (session/prompt)."""
        logger.info("Received prompt request for session: %s", session_id)
        cancel_event = self._cancel_events.setdefault(session_id, asyncio.Event())
        cancel_event.clear()
        self._store_user_prompt(session_id, prompt)
        history = build_chat_history(self._session_history.get(session_id, []))
        # Reset last text chunk tracking for this prompt turn.
        self._session_last_chunk[session_id] = None
        model_error = self._session_model_errors.get(session_id)
        if model_error:
            return await self._respond_model_error(session_id, model_error)

        for block in prompt:
            tool_call = getattr(block, "toolCall", None)
            if tool_call:
                await self._handle_tool_call(session_id, tool_call)
                return PromptResponse(stop_reason="end_turn")

        prompt_text = extract_prompt_text(prompt)

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

        runner = self._session_models.get(session_id) or self._ai_runner
        planner = (
            self._session_planners.get(session_id) or self._planning_runner or self._ai_runner
        )
        if runner is None or planner is None:
            return await self._respond_model_error(
                session_id, self._session_model_errors.get(session_id)
            )

        # --- Planning phase (programmatic hand-off) ---
        plan_only = is_plan_only_prompt(prompt_text)
        plan_response, plan_usage = await run_with_runner(planner, prompt_text, history=history)
        if plan_response.startswith("Provider error:"):
            msg = plan_response.removeprefix("Provider error:").strip()
            await self._send_update(
                session_notification(
                    session_id,
                    update_agent_message(
                        text_block(f"Model/provider error during planning: {msg}")
                    ),
                )
            )
            return PromptResponse(stop_reason="end_turn")

        plan_update = parse_plan_from_text(plan_response or "")
        if not plan_update and plan_response:
            entries = [plan_entry(plan_response)]
            plan_update = update_plan(entries)
        if plan_update:
            logger.info(
                "Planning phase produced %s steps for session %s",
                len(plan_update.entries) if getattr(plan_update, "entries", None) else 0,
                session_id,
            )
            await self._send_update(session_notification(session_id, plan_update))
            # refresh history so execution sees the plan
            history = build_chat_history(self._session_history.get(session_id, []))
        if plan_only:
            return PromptResponse(stop_reason="end_turn")

        # --- Execution phase ---
        tool_trackers: Dict[str, ToolCallTracker] = {}
        run_command_ctx_tokens: Dict[str, Any] = {}

        async def _push_chunk(chunk: str) -> None:
            last = self._session_last_chunk.get(session_id)
            if chunk == last:
                return
            self._session_last_chunk[session_id] = chunk
            await self._send_update(
                session_notification(
                    session_id,
                    update_agent_message(text_block(chunk)),
                )
            )

        async def _handle_runner_event(event: Any) -> bool:
            if isinstance(event, FunctionToolCallEvent):
                tool_name = getattr(event.part, "tool_name", None) or ""
                raw_args = getattr(event.part, "args", None)
                args = raw_args if isinstance(raw_args, dict) else {}
                tracker = ToolCallTracker(id_factory=lambda: event.tool_call_id)
                tool_trackers[event.tool_call_id] = tracker
                logger.info(
                    "LLM requested tool %s session=%s args_keys=%s",
                    tool_name,
                    session_id,
                    sorted(args.keys()) if isinstance(args, dict) else [],
                )
                start = tracker.start(
                    external_id=event.tool_call_id,
                    title=tool_name,
                    status="in_progress",
                    raw_input={"tool": tool_name, **(args if isinstance(args, dict) else {})},
                )
                await self._send_update(session_notification(session_id, start))
                if tool_name == "run_command":
                    allowed = True
                    mode = self._session_modes.get(session_id, "ask")
                    if mode == "ask":
                        allowed = await self._request_run_permission(
                            session_id,
                            tool_call_id=event.tool_call_id,
                            command=str(args.get("command") if isinstance(args, dict) else "")
                            if args
                            else "",
                            cwd=args.get("cwd") if isinstance(args, dict) else None,
                        )
                    if not allowed:
                        denied = tracker.progress(
                            external_id=event.tool_call_id,
                            status="failed",
                            raw_output={
                                "tool": tool_name,
                                "content": None,
                                "error": "permission denied",
                            },
                            content=[
                                tool_content(text_block("Command blocked: permission denied"))
                            ],
                        )
                        await self._send_update(session_notification(session_id, denied))
                        return True

                    token = set_run_command_context(
                        RunCommandContext(request_permission=lambda *_: True)
                    )
                    run_command_ctx_tokens[event.tool_call_id] = token
                return True

            if isinstance(event, FunctionToolResultEvent):
                token = run_command_ctx_tokens.pop(event.tool_call_id, None)
                if token is not None:
                    reset_run_command_context(token)
                tracker = tool_trackers.pop(event.tool_call_id, None) or ToolCallTracker(
                    id_factory=lambda: event.tool_call_id
                )
                result_part = event.result
                tool_name = getattr(result_part, "tool_name", None) or ""
                content = getattr(result_part, "content", None)
                raw_output: dict[str, Any] = {}
                if isinstance(content, dict):
                    raw_output.update(content)
                else:
                    raw_output["content"] = content
                raw_output.setdefault("tool", tool_name)
                status = "completed"
                if isinstance(result_part, RetryPromptPart):
                    raw_output["error"] = result_part.model_response()
                    status = "failed"
                else:
                    raw_output.setdefault("error", None)
                summary = raw_output.get("error") or raw_output.get("content") or ""
                logger.info(
                    "LLM tool result %s session=%s status=%s preview=%s",
                    tool_name,
                    session_id,
                    status,
                    str(summary)[:160].replace("\n", "\\n"),
                )
                progress = tracker.progress(
                    external_id=event.tool_call_id,
                    status=status,
                    raw_output=raw_output,
                    content=[tool_content(text_block(str(summary)))] if summary else None,
                )
                await self._send_update(session_notification(session_id, progress))
                return True

            return False

        executor_prompt = prompt_text
        plan_lines = [getattr(e, "content", "") for e in getattr(plan_update, "entries", []) or []]
        if plan_lines:
            plan_block = "\n".join(f"- {line}" for line in plan_lines if line)
            executor_prompt = (
                f"{prompt_text}\n\nPlan:\n{plan_block}\n\n"
                "Execute this plan now. Use tools to make progress and report results."
            )
        elif plan_response:
            executor_prompt = (
                f"{prompt_text}\n\nPlan:\n{plan_response}\n\n"
                "Execute this plan now. Use tools to make progress and report results."
            )

        response_text, usage = await stream_with_runner(
            runner,
            executor_prompt,
            _push_chunk,
            cancel_event,
            history=history,
            on_event=_handle_runner_event,
        )
        if response_text is None:
            return PromptResponse(stop_reason="cancelled")
        if response_text.startswith("Provider error:"):
            msg = response_text.removeprefix("Provider error:").strip()
            await self._send_update(
                session_notification(
                    session_id,
                    update_agent_message(text_block(f"Model/provider error: {msg}")),
                )
            )
            return PromptResponse(stop_reason="end_turn")
        # If nothing was streamed (e.g., fallback runner), ensure the response is sent once.
        exec_plan_update = None
        if not plan_update and not plan_response:
            exec_plan_update = parse_plan_from_text(response_text or "")
        if exec_plan_update:
            logger.info(
                "Parsed plan from model text for session %s entries=%s",
                session_id,
                len(exec_plan_update.entries) if getattr(exec_plan_update, "entries", None) else 0,
            )
            await self._send_update(session_notification(session_id, exec_plan_update))
        # If nothing was streamed (e.g., fallback runner), ensure the response is sent once.
        if not response_text:
            await _push_chunk(response_text)
        combined_usage = normalize_usage(usage) or normalize_usage(plan_usage)
        self._session_usage[session_id] = combined_usage
        return PromptResponse(stop_reason="end_turn")

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
        tool_call_id = getattr(tool_call, "toolCallId", str(uuid.uuid4()))

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
        result_with_tool, was_truncated = truncate_tool_output(
            result_with_tool, self._tool_output_limit
        )
        if was_truncated:
            result_with_tool["truncated"] = True
        status = "completed" if not result_with_tool.get("error") else "failed"
        summary = result_with_tool.get("content") or result_with_tool.get("error") or ""
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
            content=[tool_content(text_block(summary))],
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
                    capped_output, chunk_truncated = truncate_text(
                        final_out.output, self._tool_output_limit
                    )
                    collected.append(capped_output)
                    truncated = truncated or final_out.truncated or chunk_truncated
                    exit_code = exit_code or (
                        final_out.exit_status.exit_code if final_out.exit_status else None
                    )

        full_output, capped = truncate_text(
            "".join(collected).rstrip("\n"), self._tool_output_limit
        )
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

    async def read_text_file(
        self, path: str, session_id: str, **kwargs: Any
    ) -> ReadTextFileResponse:
        """Serve fs/read_text_file to clients (File System section)."""
        params = ReadTextFileRequest(path=path, session_id=session_id, field_meta=kwargs or None)
        return await read_text_file(self._session_cwds, params)

    async def write_text_file(
        self, content: str, path: str, session_id: str, **kwargs: Any
    ) -> WriteTextFileResponse:
        """Serve fs/write_text_file to clients (File System section)."""
        params = WriteTextFileRequest(
            content=content, path=path, session_id=session_id, field_meta=kwargs or None
        )
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

    async def terminal_output(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> TerminalOutputResponse:
        """Stream terminal output (Terminals section)."""
        params = TerminalOutputRequest(
            session_id=session_id, terminal_id=terminal_id, field_meta=kwargs or None
        )
        return await terminal_output(self._terminals, params)

    async def wait_for_terminal_exit(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> WaitForTerminalExitResponse:
        """Wait for a terminal to exit (Terminals section)."""
        params = WaitForTerminalExitRequest(
            session_id=session_id, terminal_id=terminal_id, field_meta=kwargs or None
        )
        return await wait_for_terminal_exit(self._terminals, params)

    async def kill_terminal_command(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> KillTerminalCommandResponse:
        """Kill a running terminal command (Terminals section)."""
        params = KillTerminalCommandRequest(
            session_id=session_id, terminal_id=terminal_id, field_meta=kwargs or None
        )
        return await kill_terminal(self._terminals, params)

    async def release_terminal(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> ReleaseTerminalResponse:
        """Release resources for a terminal (Terminals section)."""
        params = ReleaseTerminalRequest(
            session_id=session_id, terminal_id=terminal_id, field_meta=kwargs or None
        )
        return await release_terminal(self._terminals, params)

    async def _send_update(self, note: SessionNotification) -> None:
        """Record and emit a session/update notification for replay support."""
        self._record_update(note)
        if self._conn is None:
            raise RuntimeError("Connection not established")
        # Support both snake_case (tests) and camelCase (ACP connection).
        sender = getattr(self._conn, "session_update", None)
        if sender is None:
            sender = getattr(self._conn, "sessionUpdate", None)
            if sender is not None:
                await sender(note)
                return
        if sender is None:
            raise RuntimeError("Connection missing session_update/sessionUpdate handler")
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

    def _current_model_id(self) -> str:
        cfg = model_registry.load_models_config()
        return cfg.get("current", "test")

    async def _init_session_runners(
        self, session_id: str, toolsets: list[Any] | None = None
    ) -> None:
        """Initialize runners for a session or surface build failures."""
        try:
            if self._custom_runners:
                executor = self._ai_runner
                planner = self._planning_runner or self._ai_runner
            else:
                executor, planner = create_default_runners(toolsets)
            self._session_models[session_id] = executor
            self._session_planners[session_id] = planner
            self._session_model_ids[session_id] = self._current_model_id()
            self._session_model_errors.pop(session_id, None)
            self._session_model_error_notified.discard(session_id)
        except Exception as exc:
            msg = f"Model load failed: {exc}"
            logger.error("Failed to build model for session %s: %s", session_id, exc)
            self._session_models[session_id] = None
            self._session_planners[session_id] = None
            self._session_model_errors[session_id] = msg
            self._session_model_error_notified.add(session_id)
            await self._send_update(
                session_notification(
                    session_id,
                    update_agent_message(text_block(msg)),
                )
            )

    async def ext_method(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Handle extension methods for model listing/selection."""
        if name == "model/list":
            session_id = payload.get("session_id") or payload.get("sessionId")
            current = self._session_model_ids.get(session_id, self._current_model_id())
            models = model_registry.list_user_models()
            return {
                "current": current,
                "models": [
                    {"id": mid, "description": meta.get("description", "")}
                    for mid, meta in models.items()
                ],
            }
        if name == "model/set":
            session_id = payload.get("session_id") or payload.get("sessionId")
            model_id = payload.get("model_id") or payload.get("modelId")
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
        self._session_store.persist_meta(
            session_id, cwd, mcp_servers or [], current_mode=current_mode
        )

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

    async def _respond_model_error(self, session_id: str, message: str | None) -> PromptResponse:
        """Emit a model error message (once) and end the prompt turn."""
        msg = message or "Model unavailable for this session."
        if session_id not in self._session_model_error_notified:
            await self._send_update(
                session_notification(
                    session_id,
                    update_agent_message(text_block(msg)),
                )
            )
            self._session_model_error_notified.add(session_id)
        # ACP stop_reason is a closed set; use refusal to signal “can’t proceed”.
        return PromptResponse(stop_reason="refusal")

    async def _send_available_commands(self, session_id: str) -> None:
        """Advertise slash commands using `available_commands_update` per ACP spec."""
        if session_id in self._session_commands_advertised:
            return
        commands = available_slash_commands()
        update = AvailableCommandsUpdate(
            session_update="available_commands_update", available_commands=commands
        )
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
                PermissionOption(
                    option_id="allow_always", name="Allow this command", kind="allow_always"
                ),
                PermissionOption(option_id="reject_once", name="Reject", kind="reject_once"),
            ]
            tool_call = ToolCall(
                toolCallId=tool_call_id,
                title="run_command",
                kind="execute",
                rawInput={"tool": "run_command", "command": command, "cwd": cwd},
                status="pending",
            )
            from acp.schema import ToolCallUpdate  # type: ignore

            tool_call_update = ToolCallUpdate.model_validate(
                getattr(tool_call, "model_dump", lambda **_: tool_call)(by_alias=True)
            )
            requester = getattr(self._conn, "request_permission", None)
            if requester is None:
                requester = getattr(self._conn, "requestPermission", None)
                if requester is not None:
                    req = RequestPermissionRequest(
                        options=options,
                        sessionId=session_id,
                        toolCall=tool_call_update,
                    )
                    resp = await requester(req)
                else:
                    raise RuntimeError("Connection missing request_permission/requestPermission")
            else:
                resp = await requester(
                    options=options, session_id=session_id, tool_call=tool_call_update
                )
            outcome = getattr(resp, "outcome", None)
            option_id = ""
            if outcome is not None:
                option_id = getattr(outcome, "option_id", "") or getattr(outcome, "optionId", "")
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
