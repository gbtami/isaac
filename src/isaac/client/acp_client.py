"""Protocol-facing ACP client implementation and session update handling."""

from __future__ import annotations

from typing import Any

import logging
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
    TerminalOutputRequest,
    TerminalOutputResponse,
    WaitForTerminalExitRequest,
    WaitForTerminalExitResponse,
)
from acp.schema import (
    AgentMessageChunk,
    AgentThoughtChunk,
    AgentPlanUpdate,
    AllowedOutcome,
    AudioContentBlock,
    DeniedOutcome,
    FileEditToolCallContent,
    CurrentModeUpdate,
    EmbeddedResourceContentBlock,
    AvailableCommandsUpdate,
    ImageContentBlock,
    ResourceContentBlock,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
)

from isaac.client.client_terminal import ClientTerminalManager
from isaac.client.display import (
    print_agent_text,
    print_diff,
    print_mode_update,
    print_plan,
    print_file_edit_diff,
    print_thought,
    print_tool,
)
from isaac.client.session_state import SessionUIState


class ACPClient(Client):
    """Minimal ACP client for exercising the protocol endpoints."""

    def __init__(self, state: SessionUIState) -> None:
        self._last_prompt = ""
        self._state = state
        self._terminal_manager = ClientTerminalManager()
        self._pending_newline = False
        self._pending_newline = False
        self._logger = logging.getLogger("acp_client")

    async def request_permission(
        self,
        options,
        session_id: str,
        tool_call: Any,
        **_: Any,
    ) -> RequestPermissionResponse:
        """Prompt the user for a permission choice (Prompt Turn permission flow)."""
        if self._state.cancel_requested:
            return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))
        raw_input = getattr(tool_call, "raw_input", None) or {}
        if raw_input.get("tool") == "run_command" and raw_input.get("command"):
            cwd = raw_input.get("cwd")
            location = f" (cwd={cwd})" if cwd else ""
            print(f"[permission] run_command: {raw_input['command']}{location}")
        self._logger.info(
            "permission.request session=%s tool=%s options=%s raw=%s",
            session_id,
            getattr(tool_call, "title", "") or getattr(tool_call, "tool_call_id", ""),
            [getattr(opt, "option_id", "<id>") for opt in options],
            raw_input,
        )
        try:
            for idx, opt in enumerate(options, start=1):
                label = getattr(opt, "label", opt.option_id)
                print(f"{idx}) {label}")
            choice = input("Permission choice (number): ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(options):
                selection = options[int(choice) - 1].option_id
            else:
                selection = options[0].option_id if options else "default"
        except Exception:
            selection = options[0].option_id if options else "default"
        self._logger.info("permission.response session=%s selection=%s", session_id, selection)
        return RequestPermissionResponse(
            outcome=AllowedOutcome(option_id=selection, outcome="selected")
        )

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Handle extension methods from the agent (unused in example client)."""
        return {}

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        """Handle extension notifications from the agent (noop for example client)."""
        return None

    def on_connect(self, *_: Any, **__: Any) -> None:
        """No-op connect hook for compatibility with ACP client interface."""
        return None

    async def write_text_file(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        raise RequestError.method_not_found("fs/write_text_file")

    async def read_text_file(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        raise RequestError.method_not_found("fs/read_text_file")

    async def create_terminal(
        self,
        command: str,
        session_id: str,
        args=None,
        cwd=None,
        env=None,
        output_byte_limit=None,
        **_: Any,
    ) -> CreateTerminalResponse:
        """Create a terminal on the client host per Terminals section."""
        req = CreateTerminalRequest(
            command=command,
            session_id=session_id,
            args=args,
            cwd=cwd,
            env=env,
            output_byte_limit=output_byte_limit,
        )
        return await self._terminal_manager.create_terminal(req)

    async def terminal_output(
        self, session_id: str, terminal_id: str, **_: Any
    ) -> TerminalOutputResponse:
        """Return terminal output to the agent."""
        req = TerminalOutputRequest(session_id=session_id, terminal_id=terminal_id)
        return await self._terminal_manager.terminal_output(req)

    async def release_terminal(
        self, session_id: str, terminal_id: str, **_: Any
    ) -> ReleaseTerminalResponse:
        req = ReleaseTerminalRequest(session_id=session_id, terminal_id=terminal_id)
        return await self._terminal_manager.release_terminal(req)

    async def wait_for_terminal_exit(
        self, session_id: str, terminal_id: str, **_: Any
    ) -> WaitForTerminalExitResponse:
        """Block until the requested client terminal exits."""
        req = WaitForTerminalExitRequest(session_id=session_id, terminal_id=terminal_id)
        return await self._terminal_manager.wait_for_terminal_exit(req)

    async def kill_terminal_command(
        self, session_id: str, terminal_id: str, **_: Any
    ) -> KillTerminalCommandResponse:
        req = KillTerminalCommandRequest(session_id=session_id, terminal_id=terminal_id)
        return await self._terminal_manager.kill_terminal(req)

    async def kill_terminal(self, *args: Any, **kwargs: Any) -> KillTerminalCommandResponse:
        """Alias for kill_terminal_command to satisfy ACP interface expectations."""
        return await self.kill_terminal_command(*args, **kwargs)

    async def session_update(
        self, session_id: str, update: SessionNotification | Any, **_: Any
    ) -> None:
        update_obj = update if not isinstance(update, SessionNotification) else update.update
        update = update_obj or update
        if isinstance(update, CurrentModeUpdate):
            self._state.current_mode = update.current_mode_id
            print_mode_update(self._state.current_mode)
            return
        if isinstance(update, AvailableCommandsUpdate):
            try:
                from isaac.client.slash import SLASH_HANDLERS

                local_slashes = set(SLASH_HANDLERS.keys())
            except Exception:
                local_slashes = set()
            cmds = {}
            for cmd in update.available_commands or []:
                name = f"/{cmd.name}"
                if name in local_slashes:
                    continue
                desc = cmd.description or ""
                hint = ""
                try:
                    hint = getattr(cmd.input.root, "hint", "") if cmd.input else ""
                except Exception:
                    hint = ""
                if hint and not desc:
                    desc = hint
                cmds[name] = desc or hint or ""
            self._state.available_agent_commands = cmds
            return
        if isinstance(update, ToolCallStart):
            if self._state.pending_newline:
                print()
                self._state.pending_newline = False
            title = getattr(update, "title", "")
            raw_in = getattr(update, "raw_input", {}) or {}
            cmd = None
            if raw_in.get("tool") == "run_command":
                cmd = raw_in.get("command")
            suffix = f" cmd=`{cmd}`" if cmd else ""
            print_tool("start", f"{title}{suffix}")
            return
        if isinstance(update, ToolCallProgress):
            if self._state.pending_newline:
                print()
                self._state.pending_newline = False
            raw_out = getattr(update, "raw_output", {}) or {}
            for block in update.content or []:
                if (
                    isinstance(block, FileEditToolCallContent)
                    or getattr(block, "type", "") == "diff"
                ):
                    try:
                        print_file_edit_diff(
                            getattr(block, "path", "") or "",
                            getattr(block, "old_text", None),
                            getattr(block, "new_text", ""),
                        )
                    except Exception:
                        pass
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
                        payload = str(inner.text)
                        if raw_out.get("tool") == "edit_file" and raw_out.get("diff"):
                            print_diff(raw_out["diff"])
                        else:
                            print_agent_text(payload)
                        self._state.pending_newline = True
            return
        if isinstance(update, AgentPlanUpdate):
            print_plan(update.entries or [])
            return
        if isinstance(update, AgentThoughtChunk):
            if not self._state.show_thinking:
                return
            content = getattr(update, "content", None)
            if content is not None and getattr(content, "text", None):
                if self._state.pending_newline:
                    print()
                    self._state.pending_newline = False
                print_thought(content.text)
                self._state.pending_newline = True
            return
        if not isinstance(update, AgentMessageChunk):
            return

        content = update.content
        display_text: str
        prefix = ""
        if isinstance(content, TextContentBlock):
            display_text = content.text
            if self._state.collect_models:
                self._state.model_buffer = (self._state.model_buffer or []) + [display_text]
            print_agent_text(display_text)
            self._state.pending_newline = True
            return
        elif isinstance(content, ImageContentBlock):
            display_text = "<image>"
        elif isinstance(content, AudioContentBlock):
            display_text = "<audio>"
        elif isinstance(content, ResourceContentBlock):
            display_text = content.uri or "<resource>"
        elif isinstance(content, EmbeddedResourceContentBlock):
            display_text = "<resource>"
        else:
            display_text = "<content>"
        print_agent_text(f"{prefix} {display_text}" if prefix else display_text)


async def set_mode(
    conn: Any,
    session_id: str,
    state: SessionUIState,
    mode: str,
) -> None:
    await conn.set_session_mode(mode_id=mode, session_id=session_id)
    state.current_mode = mode
    print_mode_update(state.current_mode)
