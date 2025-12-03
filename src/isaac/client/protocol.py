"""Protocol-facing ACP client implementation and session update handling."""

from __future__ import annotations

from typing import Any

from acp import (
    Client,
    CreateTerminalRequest,
    CreateTerminalResponse,
    KillTerminalCommandRequest,
    KillTerminalCommandResponse,
    ReleaseTerminalRequest,
    ReleaseTerminalResponse,
    RequestError,
    RequestPermissionResponse,
    SessionNotification,
    SetSessionModeRequest,
    TerminalOutputRequest,
    TerminalOutputResponse,
    WaitForTerminalExitRequest,
    WaitForTerminalExitResponse,
)
from acp.schema import (
    AgentMessageChunk,
    AgentPlanUpdate,
    AllowedOutcome,
    AudioContentBlock,
    CurrentModeUpdate,
    EmbeddedResourceContentBlock,
    ImageContentBlock,
    ResourceContentBlock,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
)

from isaac.client.client_terminal import ClientTerminalManager
from isaac.client.display import print_agent_text, print_mode_update, print_plan, print_tool
from isaac.client.session_state import SessionUIState


class ExampleClient(Client):
    """Minimal ACP client for exercising the protocol endpoints."""

    def __init__(self, state: SessionUIState) -> None:
        self._last_prompt = ""
        self._state = state
        self._terminal_manager = ClientTerminalManager()

    async def requestPermission(self, params):  # type: ignore[override]
        """Prompt the user for a permission choice (Prompt Turn permission flow)."""
        try:
            for idx, opt in enumerate(params.options, start=1):
                label = getattr(opt, "label", opt.optionId)
                print(f"{idx}) {label}")
            choice = input("Permission choice (number): ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(params.options):
                selection = params.options[int(choice) - 1].optionId
            else:
                selection = params.options[0].optionId if params.options else "default"
        except Exception:
            selection = params.options[0].optionId if params.options else "default"
        return RequestPermissionResponse(outcome=AllowedOutcome(optionId=selection))

    async def writeTextFile(self, params):  # type: ignore[override]
        raise RequestError.method_not_found("fs/write_text_file")

    async def readTextFile(self, params):  # type: ignore[override]
        raise RequestError.method_not_found("fs/read_text_file")

    async def createTerminal(self, params: CreateTerminalRequest) -> CreateTerminalResponse:  # type: ignore[override]
        """Create a terminal on the client host per Terminals section."""
        return await self._terminal_manager.create_terminal(params)

    async def terminalOutput(self, params: TerminalOutputRequest) -> TerminalOutputResponse:  # type: ignore[override]
        """Return terminal output to the agent."""
        return await self._terminal_manager.terminal_output(params)

    async def releaseTerminal(self, params: ReleaseTerminalRequest) -> ReleaseTerminalResponse:  # type: ignore[override]
        return await self._terminal_manager.release_terminal(params)

    async def waitForTerminalExit(
        self, params: WaitForTerminalExitRequest
    ) -> WaitForTerminalExitResponse:  # type: ignore[override]
        """Block until the requested client terminal exits."""
        return await self._terminal_manager.wait_for_terminal_exit(params)

    async def killTerminalCommand(
        self, params: KillTerminalCommandRequest
    ) -> KillTerminalCommandResponse:  # type: ignore[override]
        return await self._terminal_manager.kill_terminal(params)

    async def sessionUpdate(self, params: SessionNotification) -> None:
        update = params.update
        if isinstance(update, CurrentModeUpdate):
            self._state.current_mode = update.currentModeId
            print_mode_update(self._state.current_mode)
            return
        if isinstance(update, ToolCallStart):
            if self._state.pending_newline:
                print()
                self._state.pending_newline = False
            print_tool("start", getattr(update, "title", ""))
            return
        if isinstance(update, ToolCallProgress):
            if self._state.pending_newline:
                print()
                self._state.pending_newline = False
            raw_out = getattr(update, "rawOutput", {}) or {}
            tool_name = raw_out.get("tool") or raw_out.get("tool_name") or getattr(
                update, "title", ""
            )
            if tool_name == "tool_generate_plan":
                # Plan is emitted via AgentPlanUpdate; avoid duplicate display here.
                return
            if update.status == "completed":
                rc = raw_out.get("returncode")
                err = raw_out.get("error")
                truncated = raw_out.get("truncated")
                summary_bits = []
                if rc is not None:
                    summary_bits.append(f"rc={rc}")
                if err:
                    summary_bits.append(f"error={err}")
                if truncated:
                    summary_bits.append("truncated")
                summary = " ".join(summary_bits) if summary_bits else "done"
                print_tool(update.status, summary)
            else:
                text = raw_out.get("content") or raw_out.get("error") or ""
                print_tool(update.status, text)
            if getattr(update, "content", None):
                for item in update.content or []:
                    inner = getattr(item, "content", None)
                    if hasattr(inner, "text") and getattr(inner, "text", None):
                        print_agent_text(str(inner.text))
                        self._state.pending_newline = True
            return
        if isinstance(update, AgentPlanUpdate):
            entries = [getattr(e, "content", "") for e in update.entries or []]
            print_plan(entries)
            return
        if not isinstance(update, AgentMessageChunk):
            return

        content = update.content
        text: str
        prefix = ""
        if isinstance(content, TextContentBlock):
            text = content.text
            if _maybe_capture_usage(text, self._state):
                return
            if self._state.collect_models:
                self._state.model_buffer = (self._state.model_buffer or []) + [text]
            print_agent_text(text)
            self._state.pending_newline = True
            return
        elif isinstance(content, ImageContentBlock):
            text = "<image>"
        elif isinstance(content, AudioContentBlock):
            text = "<audio>"
        elif isinstance(content, ResourceContentBlock):
            text = content.uri or "<resource>"
        elif isinstance(content, EmbeddedResourceContentBlock):
            text = "<resource>"
        else:
            text = "<content>"
        print_agent_text(f"{prefix} {text}" if prefix else text)


def _maybe_capture_usage(text: str, state: SessionUIState) -> bool:
    """Capture usage marker sent by agent and store in state."""
    if not text.startswith("[usage]"):
        return False
    parts = text.split()
    kv = {}
    for part in parts[1:]:
        if "=" in part:
            k, v = part.split("=", 1)
            kv[k] = v
    pct = kv.get("pct")
    if pct:
        pct_val = pct.rstrip("%")
        try:
            state.usage_summary = f"{float(pct_val):.0f}% left"
        except Exception:
            state.usage_summary = f"{pct} left"
    else:
        state.usage_summary = None
    return True


def _extract_plan_entries(text: str) -> list[str]:
    """Parse simple plan text into entries."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []
    start_idx = None
    for idx, line in enumerate(lines):
        if line.lower().startswith("plan:"):
            start_idx = idx + 1
            break
    if start_idx is None or start_idx >= len(lines):
        return []
    entries: list[str] = []
    for line in lines[start_idx:]:
        if line[0] in {"-", "*"}:
            entries.append(line.lstrip("-* ").strip())
        elif line[:2].isdigit() and "." in line:
            entries.append(line.split(".", 1)[1].strip())
    return entries


async def set_mode(
    conn: Any,
    session_id: str,
    state: SessionUIState,
    mode: str,
) -> None:
    await conn.setSessionMode(SetSessionModeRequest(sessionId=session_id, modeId=mode))
    state.current_mode = mode
    print_mode_update(state.current_mode)
