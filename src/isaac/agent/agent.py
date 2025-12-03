"""ACP agent that satisfies the Agent Client Protocol spec end-to-end.

Key ACP sections covered here:
- Initialization/version negotiation: https://agentclientprotocol.com/protocol/initialization
- Session lifecycle (new/load/prompt/cancel): https://agentclientprotocol.com/protocol/session-setup
- Prompt turn handling: https://agentclientprotocol.com/protocol/prompt-turn
- Tool calls and plan updates: https://agentclientprotocol.com/protocol/tool-calls
- File system access: https://agentclientprotocol.com/protocol/file-system
- Terminals: https://agentclientprotocol.com/protocol/terminals
- Session modes and slash commands: https://agentclientprotocol.com/protocol/session-modes
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

from acp import (
    Agent,
    AgentSideConnection,
    AuthenticateRequest,
    AuthenticateResponse,
    CancelNotification,
    InitializeRequest,
    InitializeResponse,
    LoadSessionRequest,
    LoadSessionResponse,
    NewSessionRequest,
    NewSessionResponse,
    PromptRequest,
    PromptResponse,
    ReadTextFileRequest,
    ReadTextFileResponse,
    SetSessionModeRequest,
    SetSessionModeResponse,
    WriteTextFileRequest,
    WriteTextFileResponse,
    RequestPermissionRequest,
    RequestPermissionResponse,
    SetSessionModelRequest,
    SetSessionModelResponse,
    CreateTerminalRequest,
    CreateTerminalResponse,
    TerminalOutputRequest,
    TerminalOutputResponse,
    WaitForTerminalExitRequest,
    WaitForTerminalExitResponse,
    KillTerminalCommandRequest,
    KillTerminalCommandResponse,
    ReleaseTerminalRequest,
    ReleaseTerminalResponse,
    stdio_streams,
    PROTOCOL_VERSION,
)
from acp.helpers import (
    session_notification,
    text_block,
    tool_content,
    update_agent_message,
)
from acp.schema import (
    AgentCapabilities,
    AllowedOutcome,
    CurrentModeUpdate,
    Implementation,
    McpCapabilities,
    PromptCapabilities,
    PermissionOption,
    SessionNotification,
    ToolCall,
    UserMessageChunk,
)

from isaac.agent.mcp_support import build_mcp_toolsets
from isaac.agent.session_store import SessionStore
from isaac.agent.tools import get_tools, parse_tool_request, run_tool
from isaac.agent.fs import read_text_file, write_text_file
from isaac.agent.agent_terminal import (
    TerminalState,
    create_terminal,
    terminal_output,
    wait_for_terminal_exit,
    kill_terminal,
    release_terminal,
)
from isaac.agent.brain.history import build_chat_history
from isaac.agent.planner import parse_plan_request, build_plan_notification
from isaac.agent.session_modes import build_mode_state
from isaac.agent.slash import handle_slash_command
from isaac.agent import models as model_registry
from isaac.agent.runner import register_tools, run_with_runner, stream_with_runner
from isaac.agent.brain.planner import parse_plan_from_text
from acp.contrib.tool_calls import ToolCallTracker
from acp.helpers import plan_entry, update_plan
from pydantic_ai.messages import FunctionToolCallEvent, FunctionToolResultEvent, RetryPromptPart

logger = logging.getLogger("acp_server")


@dataclass
class SimpleRunResult:
    output: str


class SimpleAIRunner:
    async def run(self, prompt: str) -> SimpleRunResult:
        return SimpleRunResult(output=f"Echo: {prompt}")

    async def run_stream_events(self, prompt: str):
        yield f"Echo: {prompt}"


def create_default_runners(toolsets: list[Any] | None = None) -> tuple[Any, Any]:
    try:
        config = model_registry.load_models_config()
        current = config.get("current", "test")
        return model_registry.build_agent_pair(current, register_tools, toolsets=toolsets or [])
    except Exception as exc:  # pragma: no cover - fallback when model creation fails
        logger.warning("Falling back to simple runner: %s", exc)
        runner = SimpleAIRunner()
        return runner, runner


class ACPAgent(Agent):
    def __init__(
        self,
        conn: AgentSideConnection,
        *,
        agent_name: str = "isaac",
        agent_title: str = "Isaac ACP Agent",
        agent_version: str = "0.1.0",
        ai_runner: Any | None = None,
        planning_runner: Any | None = None,
    ) -> None:
        self._conn = conn
        self._sessions: set[str] = set()
        self._session_cwds: dict[str, Path] = {}
        self._terminals: Dict[str, TerminalState] = {}
        self._session_modes: Dict[str, str] = {}
        self._cancel_events: Dict[str, asyncio.Event] = {}
        self._session_models: Dict[str, Any] = {}
        self._session_planners: Dict[str, Any] = {}
        self._session_model_ids: Dict[str, str] = {}
        self._session_history: Dict[str, list[SessionNotification]] = {}
        self._session_allowed_commands: Dict[str, set[tuple[str, str]]] = {}
        self._session_mcp_servers: Dict[str, Any] = {}
        self._session_toolsets: Dict[str, list[Any]] = {}
        self._agent_name = agent_name
        self._agent_title = agent_title
        self._agent_version = agent_version
        if ai_runner is None and planning_runner is None:
            self._ai_runner, self._planning_runner = create_default_runners()
        else:
            self._ai_runner = ai_runner or SimpleAIRunner()
            self._planning_runner = planning_runner or SimpleAIRunner()
        self._command_timeout_s = 30.0
        self._terminal_output_limit = 64 * 1024  # 64KB cap for streamed tool output
        self._session_store = SessionStore(Path.home() / ".isaac" / "sessions")
        self._session_last_chunk: Dict[str, str | None] = {}

    async def initialize(self, params: InitializeRequest) -> InitializeResponse:
        """Handle ACP initialize handshake (Initialization section)."""
        logger.info("Received initialize request: %s", params)
        capabilities = AgentCapabilities(
            loadSession=True,
            promptCapabilities=PromptCapabilities(
                embeddedContext=True,
                image=False,
                audio=False,
            ),
            mcpCapabilities=McpCapabilities(http=True, sse=True),
        )
        capabilities.field_meta = {"extMethods": ["model/list", "model/set"]}
        try:
            capabilities.tools = get_tools()  # type: ignore[attr-defined]
        except Exception:
            pass

        return InitializeResponse(
            protocolVersion=PROTOCOL_VERSION,
            agentCapabilities=capabilities,
            agentInfo=Implementation(
                name=self._agent_name,
                title=self._agent_title,
                version=self._agent_version,
            ),
        )

    async def authenticate(self, params: AuthenticateRequest) -> AuthenticateResponse | None:
        """Return a no-op authentication response (Initialization auth step)."""
        logger.info("Received authenticate request %s", params.methodId)
        return AuthenticateResponse()

    async def requestPermission(
        self, params: RequestPermissionRequest
    ) -> RequestPermissionResponse:
        """Return a permission outcome, following Prompt Turn guidance for gated tools."""
        # Auto-select the first available option; can be extended for richer policies.
        option_id = params.options[0].optionId if params.options else "default"
        return RequestPermissionResponse(outcome=AllowedOutcome(optionId=option_id))

    async def newSession(self, params: NewSessionRequest) -> NewSessionResponse:
        """Create a new session per Session Setup (tracks cwd, modes, and history)."""
        logger.info("Received new session request")
        self._session_store.cleanup()
        session_id = str(uuid.uuid4())
        self._sessions.add(session_id)
        cwd = Path(params.cwd or Path.cwd()).expanduser()
        if not cwd.is_absolute():
            cwd = (Path.cwd() / cwd).resolve()
        self._session_cwds[session_id] = cwd
        self._cancel_events[session_id] = asyncio.Event()
        toolsets = build_mcp_toolsets(params.mcpServers)
        self._session_toolsets[session_id] = toolsets
        executor, planner = create_default_runners(toolsets)
        self._session_models[session_id] = executor
        self._session_planners[session_id] = planner
        self._session_model_ids[session_id] = self._current_model_id()
        self._session_history[session_id] = []
        self._session_allowed_commands[session_id] = set()
        self._session_mcp_servers[session_id] = params.mcpServers
        self._persist_session_meta(session_id, cwd, params.mcpServers, current_mode="ask")
        mode_state = build_mode_state(self._session_modes, session_id, current_mode="ask")
        await self._send_usage_hint(session_id)
        return NewSessionResponse(sessionId=session_id, modes=mode_state)

    async def loadSession(self, params: LoadSessionRequest) -> LoadSessionResponse | None:
        """Reload an existing session and replay history (Session Setup / loading)."""
        logger.info("Received load session request %s", params.sessionId)
        self._sessions.add(params.sessionId)
        stored_meta = self._load_session_meta(params.sessionId)
        cwd = Path(params.cwd or stored_meta.get("cwd") or Path.cwd()).expanduser()
        if not cwd.is_absolute():
            cwd = (Path.cwd() / cwd).resolve()
        self._session_cwds[params.sessionId] = cwd
        self._cancel_events.setdefault(params.sessionId, asyncio.Event())
        mcp_servers = params.mcpServers or stored_meta.get("mcpServers", [])
        toolsets = build_mcp_toolsets(mcp_servers)
        self._session_toolsets[params.sessionId] = toolsets
        executor, planner = create_default_runners(toolsets)
        self._session_models[params.sessionId] = executor
        self._session_planners[params.sessionId] = planner
        self._session_model_ids[params.sessionId] = self._current_model_id()
        self._session_history.setdefault(params.sessionId, [])
        self._session_allowed_commands.setdefault(params.sessionId, set())
        self._session_mcp_servers[params.sessionId] = mcp_servers
        mode = stored_meta.get("mode") or self._session_modes.get(params.sessionId, "ask")
        self._session_modes[params.sessionId] = mode
        history = self._load_session_history(params.sessionId)
        self._session_history[params.sessionId].extend(history)
        self._persist_session_meta(params.sessionId, cwd, mcp_servers, current_mode=mode)
        for note in history:
            await self._conn.sessionUpdate(note)
        await self._send_usage_hint(params.sessionId)
        return LoadSessionResponse()

    async def setSessionMode(self, params: SetSessionModeRequest) -> SetSessionModeResponse | None:
        """Update the current session mode and broadcast (Session Modes)."""
        logger.info(
            "Received set session mode request %s -> %s",
            params.sessionId,
            params.modeId,
        )
        self._session_modes[params.sessionId] = params.modeId
        await self._send_update(
            session_notification(
                params.sessionId,
                CurrentModeUpdate(sessionUpdate="current_mode_update", currentModeId=params.modeId),
            )
        )
        self._persist_session_meta(
            params.sessionId,
            self._session_cwds.get(params.sessionId, Path.cwd()),
            self._session_mcp_servers.get(params.sessionId, []),
            current_mode=params.modeId,
        )
        return SetSessionModeResponse()

    async def setSessionModel(
        self, params: SetSessionModelRequest
    ) -> SetSessionModelResponse | None:
        """Switch the backing model for a session."""
        logger.info("Received set session model request %s -> %s", params.sessionId, params.modelId)
        try:
            toolsets = self._session_toolsets.get(params.sessionId, [])
            executor, planner = model_registry.build_agent_pair(
                params.modelId,
                register_tools,
                toolsets=toolsets,
            )
            self._session_models[params.sessionId] = executor
            self._session_planners[params.sessionId] = planner
            self._session_model_ids[params.sessionId] = params.modelId
            with contextlib.suppress(Exception):
                # Persist selection so subsequent runs default to the same model.
                model_registry.set_current_model(params.modelId)
            await self._send_usage_hint(params.sessionId)
        except Exception as exc:  # pragma: no cover - model build errors
            logger.error("Failed to set session model: %s", exc)
            return SetSessionModelResponse()
        return SetSessionModelResponse()

    async def prompt(self, params: PromptRequest) -> PromptResponse:
        """Process a prompt turn per Prompt Turn lifecycle (session/prompt)."""
        logger.info("Received prompt request for session: %s", params.sessionId)
        cancel_event = self._cancel_events.setdefault(params.sessionId, asyncio.Event())
        cancel_event.clear()
        self._store_user_prompt(params.sessionId, params.prompt)
        history = build_chat_history(self._session_history.get(params.sessionId, []))
        # Reset last text chunk tracking for this prompt turn.
        self._session_last_chunk[params.sessionId] = None

        for block in params.prompt:
            tool_call = getattr(block, "toolCall", None)
            if tool_call:
                await self._handle_tool_call(params.sessionId, tool_call)
                return PromptResponse(stopReason="end_turn")

        prompt_text = _extract_prompt_text(params.prompt)

        slash = handle_slash_command(params.sessionId, prompt_text)
        if slash:
            await self._send_update(slash)
            return PromptResponse(stopReason="end_turn")

        plan_request = parse_plan_request(prompt_text)
        if plan_request:
            await self._send_update(build_plan_notification(params.sessionId, plan_request))
            return PromptResponse(stopReason="end_turn")

        if prompt_text.startswith("/model"):
            note = self._handle_model_command(params.sessionId, prompt_text)
            if note:
                await self._send_update(note)
            return PromptResponse(stopReason="end_turn")

        tool_request = parse_tool_request(prompt_text)
        if tool_request:
            tool_call = SimpleNamespace(
                toolCallId=str(uuid.uuid4()),
                function=tool_request.pop("tool_name"),
                arguments=tool_request,
            )
            await self._handle_tool_call(params.sessionId, tool_call)
            return PromptResponse(stopReason="end_turn")

        if cancel_event.is_set():
            return PromptResponse(stopReason="cancelled")

        runner = self._session_models.get(params.sessionId, self._ai_runner)
        planner = self._session_planners.get(params.sessionId, self._planning_runner)

        # --- Planning phase (programmatic hand-off) ---
        plan_only = _is_plan_only_prompt(prompt_text)
        plan_response, plan_usage = await run_with_runner(planner, prompt_text, history=history)
        if plan_response.startswith("Provider error:"):
            msg = plan_response.removeprefix("Provider error:").strip()
            await self._send_update(
                session_notification(
                    params.sessionId,
                    update_agent_message(
                        text_block(f"Model/provider error during planning: {msg}")
                    ),
                )
            )
            return PromptResponse(stopReason="end_turn")

        plan_update = parse_plan_from_text(plan_response or "")
        if not plan_update and plan_response:
            entries = [plan_entry(plan_response)]
            plan_update = update_plan(entries)
        if plan_update:
            logger.info(
                "Planning phase produced %s steps for session %s",
                len(plan_update.entries) if getattr(plan_update, "entries", None) else 0,
                params.sessionId,
            )
            await self._send_update(session_notification(params.sessionId, plan_update))
            # refresh history so execution sees the plan
            history = build_chat_history(self._session_history.get(params.sessionId, []))
        if plan_only:
            return PromptResponse(stopReason="end_turn")

        # --- Execution phase ---
        tool_trackers: Dict[str, ToolCallTracker] = {}

        async def _push_chunk(chunk: str) -> None:
            last = self._session_last_chunk.get(params.sessionId)
            if chunk == last:
                return
            self._session_last_chunk[params.sessionId] = chunk
            await self._send_update(
                session_notification(
                    params.sessionId,
                    update_agent_message(text_block(chunk)),
                )
            )

        async def _handle_runner_event(event: Any) -> bool:
            if isinstance(event, FunctionToolCallEvent):
                tool_name = getattr(event.part, "tool_name", None) or ""
                tracker = ToolCallTracker(id_factory=lambda: event.tool_call_id)
                tool_trackers[event.tool_call_id] = tracker
                logger.info(
                    "LLM requested tool %s session=%s args_keys=%s",
                    tool_name,
                    params.sessionId,
                    sorted((event.part.args or {}).keys()),
                )
                start = tracker.start(
                    external_id=event.tool_call_id,
                    title=tool_name,
                    status="in_progress",
                    raw_input={"tool": tool_name, **(event.part.args or {})},
                )
                await self._send_update(session_notification(params.sessionId, start))
                return True

            if isinstance(event, FunctionToolResultEvent):
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
                    params.sessionId,
                    status,
                    str(summary)[:160].replace("\n", "\\n"),
                )
                progress = tracker.progress(
                    external_id=event.tool_call_id,
                    status=status,
                    raw_output=raw_output,
                    content=[tool_content(text_block(str(summary)))] if summary else None,
                )
                await self._send_update(session_notification(params.sessionId, progress))
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
            return PromptResponse(stopReason="cancelled")
        if response_text.startswith("Provider error:"):
            msg = response_text.removeprefix("Provider error:").strip()
            await self._send_update(
                session_notification(
                    params.sessionId,
                    update_agent_message(text_block(f"Model/provider error: {msg}")),
                )
            )
            return PromptResponse(stopReason="end_turn")
        # If nothing was streamed (e.g., fallback runner), ensure the response is sent once.
        plan_update = parse_plan_from_text(response_text or "")
        if plan_update:
            logger.info(
                "Parsed plan from model text for session %s entries=%s",
                params.sessionId,
                len(plan_update.entries) if getattr(plan_update, "entries", None) else 0,
            )
            await self._send_update(session_notification(params.sessionId, plan_update))
        # If nothing was streamed (e.g., fallback runner), ensure the response is sent once.
        if not response_text:
            await _push_chunk(response_text)
        context_limit = model_registry.get_context_limit(
            self._session_model_ids.get(params.sessionId, "")
        )
        combined_usage = usage or plan_usage
        usage_text = _format_usage(
            combined_usage,
            context_limit,
            self._session_model_ids.get(params.sessionId, ""),
        )
        if not usage_text and context_limit:
            usage_text = "[usage] pct=100"
        if usage_text:
            await self._send_update(
                session_notification(params.sessionId, update_agent_message(text_block(usage_text)))
            )
        return PromptResponse(stopReason="end_turn")

    async def cancel(self, params: CancelNotification) -> None:
        """Stop in-flight prompt/tool work for a session (Prompt Turn cancellation)."""
        logger.info("Received cancel notification for session %s", params.sessionId)
        event = self._cancel_events.get(params.sessionId)
        if event:
            event.set()

    async def _handle_tool_call(self, session_id: str, tool_call: Any) -> None:
        """Dispatch a tool call coming from the model (Tool Calls section)."""
        function_name = getattr(tool_call, "function", "")
        arguments = getattr(tool_call, "arguments", {}) or {}
        tool_call_id = getattr(tool_call, "toolCallId", str(uuid.uuid4()))

        if function_name == "tool_run_command":
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
        result: dict[str, Any] | None = await _await_with_cancel(
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

        status = "completed" if not result.get("error") else "failed"
        summary = result.get("content") or result.get("error") or ""
        # Include tool name for downstream clients to interpret plan tool output.
        result_with_tool = dict(result)
        result_with_tool.setdefault("tool", tool_name)
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
            title="tool_run_command",
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
                    sessionId=session_id,
                    command="bash",
                    args=["-lc", command],
                    cwd=cwd_arg,
                    outputByteLimit=self._terminal_output_limit,
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

        term_id = create_resp.terminalId
        collected: list[str] = []
        truncated = False
        exit_code: int | None = None
        error_msg: str | None = None

        try:
            start_time = asyncio.get_event_loop().time()
            while True:
                if cancel_event.is_set():
                    error_msg = "cancelled"
                    await self.killTerminalCommand(
                        KillTerminalCommandRequest(sessionId=session_id, terminalId=term_id)
                    )
                    break

                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed > timeout_s:
                    error_msg = f"Command timed out after {timeout_s}s"
                    await self.killTerminalCommand(
                        KillTerminalCommandRequest(sessionId=session_id, terminalId=term_id)
                    )
                    break

                out_resp = await terminal_output(
                    self._terminals,
                    TerminalOutputRequest(sessionId=session_id, terminalId=term_id),
                )
                chunk = out_resp.output or ""
                if chunk:
                    collected.append(chunk)
                    truncated = truncated or out_resp.truncated
                    progress = tracker.progress(
                        external_id=tool_call_id,
                        status="in_progress",
                        raw_output={
                            "content": chunk,
                            "error": None,
                            "returncode": exit_code,
                            "truncated": out_resp.truncated,
                        },
                        content=[tool_content(text_block(chunk))],
                    )
                    await self._send_update(session_notification(session_id, progress))

                if out_resp.exitStatus:
                    exit_code = out_resp.exitStatus.exitCode
                    break

                await asyncio.sleep(0.2)
        finally:
            with contextlib.suppress(Exception):
                await release_terminal(
                    self._terminals,
                    ReleaseTerminalRequest(sessionId=session_id, terminalId=term_id),
                )

        if not collected:
            with contextlib.suppress(Exception):
                final_out = await terminal_output(
                    self._terminals,
                    TerminalOutputRequest(sessionId=session_id, terminalId=term_id),
                )
                if final_out.output:
                    collected.append(final_out.output)
                    truncated = truncated or final_out.truncated
                    exit_code = exit_code or (
                        final_out.exitStatus.exitCode if final_out.exitStatus else None
                    )

        full_output = "".join(collected).rstrip("\n")
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

    def _handle_model_command(self, session_id: str, prompt_text: str):
        """Handle `/model` control commands sent as prompt text."""
        parts = prompt_text.split()
        if len(parts) == 1:
            models = model_registry.list_user_models()
            current = self._session_model_ids.get(session_id, self._current_model_id())
            lines = [f"Current model: {current}", "Available:"]
            for mid, meta in models.items():
                desc = meta.get("description") or ""
                lines.append(f"- {mid}: {desc}")
            return session_notification(
                session_id,
                update_agent_message(text_block("\n".join(lines))),
            )

        model_id = parts[1]
        try:
            model_registry.set_current_model(model_id)
            executor, planner = model_registry.build_agent_pair(model_id, register_tools)
            self._ai_runner = executor
            self._planning_runner = planner
            msg = f"Switched to model '{model_id}'."
        except Exception as exc:
            msg = f"Failed to switch model '{model_id}': {exc}"

        return session_notification(session_id, update_agent_message(text_block(msg)))

    async def readTextFile(self, params: ReadTextFileRequest) -> ReadTextFileResponse:
        """Serve fs/read_text_file to clients (File System section)."""
        return await read_text_file(self._session_cwds, params)

    async def writeTextFile(self, params: WriteTextFileRequest) -> WriteTextFileResponse:
        """Serve fs/write_text_file to clients (File System section)."""
        return await write_text_file(self._session_cwds, params)

    async def createTerminal(self, params: CreateTerminalRequest) -> CreateTerminalResponse:
        """Create a terminal on the agent host (Terminals section)."""
        return await create_terminal(self._session_cwds, self._terminals, params)

    async def terminalOutput(self, params: TerminalOutputRequest) -> TerminalOutputResponse:
        """Stream terminal output (Terminals section)."""
        return await terminal_output(self._terminals, params)

    async def waitForTerminalExit(
        self, params: WaitForTerminalExitRequest
    ) -> WaitForTerminalExitResponse:
        """Wait for a terminal to exit (Terminals section)."""
        return await wait_for_terminal_exit(self._terminals, params)

    async def killTerminalCommand(
        self, params: KillTerminalCommandRequest
    ) -> KillTerminalCommandResponse:
        """Kill a running terminal command (Terminals section)."""
        return await kill_terminal(self._terminals, params)

    async def releaseTerminal(self, params: ReleaseTerminalRequest) -> ReleaseTerminalResponse:
        """Release resources for a terminal (Terminals section)."""
        return await release_terminal(self._terminals, params)

    async def _send_update(self, note: SessionNotification) -> None:
        """Record and emit a session/update notification for replay support."""
        self._record_update(note)
        await self._conn.sessionUpdate(note)

    def _record_update(self, note: SessionNotification) -> None:
        """Cache and persist updates for replay after restarts."""
        history = self._session_history.setdefault(note.sessionId, [])
        history.append(note)
        self._session_store.persist_update(note.sessionId, note)

    def _store_user_prompt(self, session_id: str, prompt_blocks: list[Any]) -> None:
        """Persist user prompt content for session/load replay."""
        for block in prompt_blocks:
            try:
                chunk = UserMessageChunk(content=block)
            except Exception:
                continue
            note = SessionNotification(sessionId=session_id, update=chunk)
            self._record_update(note)

    def _current_model_id(self) -> str:
        cfg = model_registry.load_models_config()
        return cfg.get("current", "test")

    async def extMethod(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Handle extension methods for model listing/selection."""
        if name == "model/list":
            session_id = payload.get("sessionId")
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
            session_id = payload.get("sessionId")
            model_id = payload.get("modelId")
            if not session_id or not model_id:
                return {"error": "sessionId and modelId required"}
            try:
                await self.setSessionModel(
                    SetSessionModelRequest(sessionId=session_id, modelId=model_id)
                )
                self._session_model_ids[session_id] = model_id
                return {"current": model_id}
            except Exception as exc:  # noqa: BLE001
                return {"error": str(exc)}
        return {"error": f"Unknown ext method: {name}"}

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

    async def _send_usage_hint(self, session_id: str) -> None:
        """Emit a usage marker so the client can show context % immediately."""
        model_id = self._session_model_ids.get(session_id, "")
        context_limit = model_registry.get_context_limit(model_id)
        if not context_limit:
            return
        usage_text = "[usage] pct=100"
        await self._send_update(
            session_notification(session_id, update_agent_message(text_block(usage_text)))
        )

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
            req = RequestPermissionRequest(
                sessionId=session_id,
                toolCall=ToolCall(
                    toolCallId=tool_call_id,
                    title="run_command",
                    kind="execute",
                    rawInput={"tool": "tool_run_command", "command": command, "cwd": cwd},
                    status="pending",
                ),
                options=[
                    PermissionOption(optionId="allow_once", name="Allow once", kind="allow_once"),
                    PermissionOption(optionId="allow_always", name="Allow this command", kind="allow_always"),
                    PermissionOption(optionId="reject_once", name="Reject", kind="reject_once"),
                ],
            )
            resp = await self._conn.requestPermission(req)
            outcome = getattr(resp, "outcome", None)
            option_id = getattr(outcome, "optionId", "") if outcome else ""
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


def _format_usage(usage: Any, context_limit: int | None, model_id: str) -> str:
    """Build a compact usage marker for the client to parse (percent remaining)."""

    def _get(field: str) -> int | None:
        if hasattr(usage, field):
            val = getattr(usage, field)
            return int(val) if isinstance(val, (int, float)) else None
        if isinstance(usage, dict) and field in usage:
            val = usage.get(field)
            return int(val) if isinstance(val, (int, float)) else None
        return None

    input_tokens = _get("input_tokens") or _get("prompt_tokens")
    if context_limit is None or input_tokens is None:
        return ""

    remaining = max(0, context_limit - input_tokens)
    pct_left = max(0.0, remaining / context_limit * 100.0)
    return f"[usage] pct={pct_left:.0f}"


async def run_acp_agent():
    """Run the ACP server."""
    _setup_acp_logging()
    logger.info("Starting ACP server on stdio")

    reader, writer = await stdio_streams()
    AgentSideConnection(lambda conn: ACPAgent(conn), writer, reader)
    await asyncio.Event().wait()


def _setup_acp_logging():
    log_dir = Path.home() / ".isaac"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "acp_server.log"

    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file)],
    )


async def main(argv: list[str] | None = None):
    """Default entrypoint launches the ACP server on stdio."""
    return await run_acp_agent()


def main_entry():
    try:
        return asyncio.run(main())
    except KeyboardInterrupt:
        return 0


async def _await_with_cancel(coro: Any, cancel_event: asyncio.Event) -> Any | None:
    wait_task = asyncio.create_task(cancel_event.wait())
    main_task = asyncio.create_task(coro)
    done, pending = await asyncio.wait(
        {main_task, wait_task},
        return_when=asyncio.FIRST_COMPLETED,
    )
    if wait_task in done and cancel_event.is_set():
        main_task.cancel()
        return None
    return main_task.result() if main_task in done else None


def _extract_prompt_text(blocks: list[Any]) -> str:
    parts: list[str] = []
    for block in blocks:
        text_val = getattr(block, "text", None)
        if text_val:
            parts.append(text_val)
            continue
        resource = getattr(block, "resource", None)
        if resource and hasattr(resource, "text") and getattr(resource, "text", None):
            parts.append(str(resource.text))
            continue
        uri = getattr(block, "uri", None)
        if uri:
            parts.append(f"[resource:{uri}]")
    return "".join(parts)


def _is_plan_only_prompt(prompt_text: str) -> bool:
    """Heuristic to avoid executing when the user explicitly asked for a plan only."""

    lowered = prompt_text.lower()
    if lowered.startswith("plan:"):
        return True
    plan_only_markers = [
        "plan only",
        "only a plan",
        "just a plan",
        "just planning",
        "planning only",
    ]
    return any(marker in lowered for marker in plan_only_markers)


if __name__ == "__main__":
    main_entry()
