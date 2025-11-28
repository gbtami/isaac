"""Core ACP agent wiring: tools, planning, filesystem, terminals, and AI runner.

Relevant ACP sections:
- Tools/tool calls: https://agentclientprotocol.com/protocol/tools
- Agent plan: https://agentclientprotocol.com/protocol/agent-plan
- File system: https://agentclientprotocol.com/protocol/file-system
- Terminals: https://agentclientprotocol.com/protocol/terminals
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import uuid
from types import SimpleNamespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict

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

from .tools import get_tools, run_tool, parse_tool_request, TOOL_HANDLERS
from .fs import read_text_file, write_text_file
from .terminal import (
    TerminalState,
    create_terminal,
    terminal_output,
    wait_for_terminal_exit,
    kill_terminal,
    release_terminal,
)
from .planner import parse_plan_request, build_plan_notification
from .session_modes import available_modes, build_mode_state
from .slash import _run_pytest, handle_slash_command
from acp.contrib.tool_calls import ToolCallTracker

logger = logging.getLogger("acp_server")

GOOGLE_API_KEY = "AIzaSyDapebATW5RB3RAL4TuN0i4PyughpUubUs"

@dataclass
class SimpleRunResult:
    output: str


class SimpleAIRunner:
    async def run(self, prompt: str) -> SimpleRunResult:
        return SimpleRunResult(output=f"Echo: {prompt}")


def _build_pydantic_runner() -> Any | None:
    try:
        from pydantic_ai import Agent as PydanticAgent  # type: ignore
        from pydantic_ai.models.openai import OpenAIChatModel  # type: ignore
        from pydantic_ai.providers.openai import OpenAIProvider  # type: ignore
    except Exception as exc:  # pragma: no cover
        logger.warning("pydantic-ai unavailable, falling back to echo runner: %s", exc)
        return None

    api_key = os.getenv("CEREBRAS_API_KEY") or os.getenv("OPENAI_API_KEY")
    if api_key:
        base_url = os.getenv("OPENAI_BASE_URL") or "https://api.cerebras.ai/v1"
        model_name = os.getenv("MODEL_NAME") or "qwen-3-235b-a22b-instruct-2507"
        provider = OpenAIProvider(base_url=base_url, api_key=api_key)
        model = OpenAIChatModel(model_name, provider=provider)
        agent = PydanticAgent(model)
    else:
        agent = PydanticAgent("test")
    print("ITT", agent)
    for name in TOOL_HANDLERS.keys():

        def _make_tool(fn_name: str):
            @agent.tool_plain(name=fn_name)  # type: ignore[misc]
            async def _wrapper(**kwargs: Any):
                return await run_tool(fn_name, **kwargs)

            return _wrapper

        _make_tool(name)

    return agent


def create_default_runner() -> Any:
    return _build_pydantic_runner() or SimpleAIRunner()


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
        mode_state = build_mode_state(self._session_modes, session_id, current_mode="ask")
        return NewSessionResponse(sessionId=session_id, modes=mode_state)

    async def loadSession(self, params: LoadSessionRequest) -> LoadSessionResponse | None:
        logger.info("Received load session request %s", params.sessionId)
        self._sessions.add(params.sessionId)
        self._session_cwds[params.sessionId] = Path.cwd()
        return LoadSessionResponse()

    async def setSessionMode(self, params: SetSessionModeRequest) -> SetSessionModeResponse | None:
        logger.info(
            "Received set session mode request %s -> %s",
            params.sessionId,
            params.modeId,
        )
        self._session_modes[params.sessionId] = params.modeId
        return SetSessionModeResponse()

    async def prompt(self, params: PromptRequest) -> PromptResponse:
        logger.info("Received prompt request for session: %s", params.sessionId)

        for block in params.prompt:
            tool_call = getattr(block, "toolCall", None)
            if tool_call:
                await self._handle_tool_call(params.sessionId, tool_call)
                return PromptResponse(stopReason="end_turn")

        prompt_text = "".join(block.text for block in params.prompt if getattr(block, "text", None))

        slash = handle_slash_command(params.sessionId, prompt_text)
        if slash:
            await self._conn.sessionUpdate(slash)
            return PromptResponse(stopReason="end_turn")

        mode = self._session_modes.get(params.sessionId, "ask")
        if mode == "reject":
            return PromptResponse(stopReason="refusal")
        if mode == "request_permission":
            await self._conn.sessionUpdate(
                session_notification(
                    params.sessionId,
                    update_agent_message(text_block("Permission required; please confirm action.")),
                )
            )
            return PromptResponse(stopReason="end_turn")
        plan_request = parse_plan_request(prompt_text)
        if plan_request:
            await self._conn.sessionUpdate(build_plan_notification(params.sessionId, plan_request))
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

        response_text = await self._run_ai(prompt_text)
        await self._conn.sessionUpdate(
            session_notification(
                params.sessionId,
                update_agent_message(text_block(response_text)),
            )
        )
        return PromptResponse(stopReason="end_turn")

    async def cancel(self, params: CancelNotification) -> None:
        logger.info("Received cancel notification for session %s", params.sessionId)

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

        result: dict[str, Any] = await run_tool(tool_name, **(arguments or {}))

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
        return await _run_with_runner(self._ai_runner, prompt_text)

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


async def main():
    parser = argparse.ArgumentParser(description="Simple ACP agent")
    parser.add_argument("--acp", action="store_true", help="Run in ACP mode")
    args = parser.parse_args()

    if args.acp:
        await run_acp_agent()
    else:
        runner = create_default_runner()
        current_mode = "ask"
        mode_ids = {m["id"] for m in available_modes()}
        approved_commands: set[str] = set()

        while True:
            try:
                prompt = input(">>> ")
                if prompt.lower() in ["exit", "quit"]:
                    break
                if prompt.strip() == "/test":
                    print(_run_pytest())
                    continue

                if prompt.startswith("/mode "):
                    requested = prompt[len("/mode ") :].strip()
                    if requested in mode_ids:
                        current_mode = requested
                        print(f"[mode set to {current_mode}]")
                    else:
                        print(
                            f"[unknown mode: {requested}; available: {', '.join(sorted(mode_ids))}]"
                        )
                    continue

                if current_mode == "reject":
                    print("[request rejected in current mode]")
                    continue
                if current_mode == "request_permission":
                    if prompt not in approved_commands:
                        try:
                            from prompt_toolkit.shortcuts import radiolist_dialog

                            result = radiolist_dialog(
                                title="Permission required",
                                text=f"Command: {prompt}",
                                values=[
                                    ("y", "Yes, proceed"),
                                    ("a", "Yes, and don't ask again for this command"),
                                    ("esc", "No, and tell me what to do differently"),
                                ],
                            ).run()
                        except Exception:
                            resp = input(
                                "Permission required. [y]es / [a]lways for this command / [esc] cancel: "
                            ).strip()
                            result = resp.lower()

                        if result == "a":
                            approved_commands.add(prompt)
                        elif result not in {"y", "yes"}:
                            print("[request cancelled] If you want me to proceed, allow the action.")
                            continue

                response_text = await _run_with_runner(runner, prompt)
                print(response_text)
            except (EOFError, KeyboardInterrupt):
                break
        print("\nExiting simple interactive agent.")


def main_entry():
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        return 0


async def _run_with_runner(runner: Any, prompt_text: str) -> str:
    run_method: Callable[[str], Any] | None = getattr(runner, "run", None)
    if not callable(run_method):
        return "Hello, world!"

    try:
        result = run_method(prompt_text)
        if asyncio.iscoroutine(result):
            result = await result

        output = getattr(result, "output", None)
        if isinstance(output, str):
            return output
        if isinstance(result, str):
            return result
    except Exception as exc:  # pragma: no cover
        logger.debug("AI runner failed, falling back to echo: %s", exc)

    return "Hello, world!"


if __name__ == "__main__":
    main_entry()
