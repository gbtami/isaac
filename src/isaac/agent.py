"""Core ACP agent wiring: tools, planning, filesystem, terminals, and AI runner.

Relevant ACP sections:
- Tools/tool calls: https://agentclientprotocol.com/protocol/tools
- Agent plan: https://agentclientprotocol.com/protocol/agent-plan
- File system: https://agentclientprotocol.com/protocol/file-system
- Terminals: https://agentclientprotocol.com/protocol/terminals
"""

from __future__ import annotations

import asyncio
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
from acp.schema import AgentCapabilities, Implementation
from acp.schema import AllowedOutcome

from isaac.tools import get_tools, parse_tool_request, run_tool
from isaac.fs import read_text_file, write_text_file
from isaac.agent_terminal import (
    TerminalState,
    create_terminal,
    terminal_output,
    wait_for_terminal_exit,
    kill_terminal,
    release_terminal,
)
from isaac.planner import parse_plan_request, build_plan_notification
from isaac.session_modes import build_mode_state
from isaac.slash import handle_slash_command
from isaac import models as model_registry
from isaac.runner import register_tools, run_with_runner, stream_with_runner
from acp.contrib.tool_calls import ToolCallTracker
logger = logging.getLogger("acp_server")

@dataclass
class SimpleRunResult:
    output: str


class SimpleAIRunner:
    async def run(self, prompt: str) -> SimpleRunResult:
        return SimpleRunResult(output=f"Echo: {prompt}")


def create_default_runner() -> Any:
    try:
        config = model_registry.load_models_config()
        current = config.get("current", "test")
        return model_registry.build_agent(current, register_tools)
    except Exception as exc:  # pragma: no cover - fallback when model creation fails
        logger.warning("Falling back to simple runner: %s", exc)
        return SimpleAIRunner()


class ACPAgent(Agent):
    def __init__(
        self,
        conn: AgentSideConnection,
        *,
        agent_name: str = "isaac",
        agent_title: str = "Isaac ACP Agent",
        agent_version: str = "0.1.0",
        ai_runner: Any | None = None,
    ) -> None:
        self._conn = conn
        self._sessions: set[str] = set()
        self._session_cwds: dict[str, Path] = {}
        self._terminals: Dict[str, TerminalState] = {}
        self._session_modes: Dict[str, str] = {}
        self._cancel_events: Dict[str, asyncio.Event] = {}
        self._session_models: Dict[str, Any] = {}
        self._agent_name = agent_name
        self._agent_title = agent_title
        self._agent_version = agent_version
        self._ai_runner = ai_runner or create_default_runner()

    async def initialize(self, params: InitializeRequest) -> InitializeResponse:
        logger.info("Received initialize request: %s", params)
        capabilities = AgentCapabilities()
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
        logger.info("Received authenticate request %s", params.methodId)
        return AuthenticateResponse()

    async def requestPermission(self, params: RequestPermissionRequest) -> RequestPermissionResponse:
        # Auto-select the first available option; can be extended for richer policies.
        option_id = params.options[0].optionId if params.options else "default"
        return RequestPermissionResponse(outcome=AllowedOutcome(optionId=option_id))

    async def newSession(self, params: NewSessionRequest) -> NewSessionResponse:
        logger.info("Received new session request")
        session_id = str(uuid.uuid4())
        self._sessions.add(session_id)
        cwd = Path(params.cwd or Path.cwd())
        self._session_cwds[session_id] = cwd
        self._cancel_events[session_id] = asyncio.Event()
        self._session_models[session_id] = create_default_runner()
        mode_state = build_mode_state(self._session_modes, session_id, current_mode="ask")
        return NewSessionResponse(sessionId=session_id, modes=mode_state)

    async def loadSession(self, params: LoadSessionRequest) -> LoadSessionResponse | None:
        logger.info("Received load session request %s", params.sessionId)
        self._sessions.add(params.sessionId)
        self._session_cwds[params.sessionId] = Path.cwd()
        self._cancel_events.setdefault(params.sessionId, asyncio.Event())
        self._session_models.setdefault(params.sessionId, create_default_runner())
        return LoadSessionResponse()

    async def setSessionMode(self, params: SetSessionModeRequest) -> SetSessionModeResponse | None:
        logger.info(
            "Received set session mode request %s -> %s",
            params.sessionId,
            params.modeId,
        )
        self._session_modes[params.sessionId] = params.modeId
        return SetSessionModeResponse()

    async def setSessionModel(self, params: SetSessionModelRequest) -> SetSessionModelResponse | None:
        logger.info("Received set session model request %s -> %s", params.sessionId, params.modelId)
        try:
            self._session_models[params.sessionId] = model_registry.build_agent(
                params.modelId,
                register_tools,
            )
        except Exception as exc:  # pragma: no cover - model build errors
            logger.error("Failed to set session model: %s", exc)
            return SetSessionModelResponse()
        return SetSessionModelResponse()

    async def prompt(self, params: PromptRequest) -> PromptResponse:
        logger.info("Received prompt request for session: %s", params.sessionId)
        cancel_event = self._cancel_events.setdefault(params.sessionId, asyncio.Event())
        cancel_event.clear()

        for block in params.prompt:
            tool_call = getattr(block, "toolCall", None)
            if tool_call:
                await self._handle_tool_call(params.sessionId, tool_call)
                return PromptResponse(stopReason="end_turn")

        prompt_text = _extract_prompt_text(params.prompt)

        slash = handle_slash_command(params.sessionId, prompt_text)
        if slash:
            await self._conn.sessionUpdate(slash)
            return PromptResponse(stopReason="end_turn")

        plan_request = parse_plan_request(prompt_text)
        if plan_request:
            await self._conn.sessionUpdate(build_plan_notification(params.sessionId, plan_request))
            return PromptResponse(stopReason="end_turn")

        if prompt_text.startswith("/model"):
            note = self._handle_model_command(params.sessionId, prompt_text)
            if note:
                await self._conn.sessionUpdate(note)
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

        async def _push_chunk(chunk: str) -> None:
            await self._conn.sessionUpdate(
                session_notification(
                    params.sessionId,
                    update_agent_message(text_block(chunk)),
                )
            )

        response_text = await stream_with_runner(runner, prompt_text, _push_chunk, cancel_event)
        if response_text is None:
            return PromptResponse(stopReason="cancelled")
        # If nothing was streamed (e.g., fallback runner), ensure the response is sent once.
        if not response_text:
            await _push_chunk(response_text)
        return PromptResponse(stopReason="end_turn")

    async def cancel(self, params: CancelNotification) -> None:
        logger.info("Received cancel notification for session %s", params.sessionId)
        event = self._cancel_events.get(params.sessionId)
        if event:
            event.set()

    async def _handle_tool_call(self, session_id: str, tool_call: Any) -> None:
        function_name = getattr(tool_call, "function", "")
        arguments = getattr(tool_call, "arguments", {}) or {}
        tool_call_id = getattr(tool_call, "toolCallId", str(uuid.uuid4()))

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
        tool_call_id = tool_call_id or str(uuid.uuid4())
        tracker = ToolCallTracker(id_factory=lambda: tool_call_id)
        start = tracker.start(
            external_id=tool_call_id,
            title=tool_name,
            status="in_progress",
            raw_input={"tool": tool_name, **(arguments or {})},
        )
        await self._conn.sessionUpdate(session_notification(session_id, start))

        cancel_event = self._cancel_events.setdefault(session_id, asyncio.Event())
        result: dict[str, Any] | None = await _await_with_cancel(
            run_tool(tool_name, **(arguments or {})),
            cancel_event,
        )
        if result is None:
            progress = tracker.progress(
                external_id=tool_call_id,
                status="failed",
                raw_output={"content": None, "error": "cancelled"},
                content=[tool_content(text_block("Cancelled"))],
            )
            await self._conn.sessionUpdate(session_notification(session_id, progress))
            return

        status = "completed" if not result.get("error") else "failed"
        summary = result.get("content") or result.get("error") or ""
        progress = tracker.progress(
            external_id=tool_call_id,
            status=status,
            raw_output=result,
            content=[tool_content(text_block(summary))],
        )
        await self._conn.sessionUpdate(session_notification(session_id, progress))

    async def _run_ai(self, prompt_text: str) -> str:
        return await run_with_runner(self._ai_runner, prompt_text)

    def _handle_model_command(self, session_id: str, prompt_text: str):
        parts = prompt_text.split()
        if len(parts) == 1:
            models = model_registry.list_user_models()
            current = model_registry.load_models_config().get("current", "test")
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
            self._ai_runner = model_registry.build_agent(model_id, register_tools)
            msg = f"Switched to model '{model_id}'."
        except Exception as exc:
            msg = f"Failed to switch model '{model_id}': {exc}"

        return session_notification(session_id, update_agent_message(text_block(msg)))

    async def readTextFile(self, params: ReadTextFileRequest) -> ReadTextFileResponse:
        return await read_text_file(self._session_cwds, params)

    async def writeTextFile(self, params: WriteTextFileRequest) -> WriteTextFileResponse:
        return await write_text_file(self._session_cwds, params)

    async def createTerminal(self, params: CreateTerminalRequest) -> CreateTerminalResponse:
        return await create_terminal(self._session_cwds, self._terminals, params)

    async def terminalOutput(self, params: TerminalOutputRequest) -> TerminalOutputResponse:
        return await terminal_output(self._terminals, params)

    async def waitForTerminalExit(
        self, params: WaitForTerminalExitRequest
    ) -> WaitForTerminalExitResponse:
        return await wait_for_terminal_exit(self._terminals, params)

    async def killTerminalCommand(
        self, params: KillTerminalCommandRequest
    ) -> KillTerminalCommandResponse:
        return await kill_terminal(self._terminals, params)

    async def releaseTerminal(self, params: ReleaseTerminalRequest) -> ReleaseTerminalResponse:
        return await release_terminal(self._terminals, params)


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
        level=logging.DEBUG,
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


if __name__ == "__main__":
    main_entry()
