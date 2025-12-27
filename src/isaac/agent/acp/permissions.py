"""Permission handling for ACP requests."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from acp import RequestPermissionResponse
from acp.helpers import text_block, tool_content
from acp.schema import PermissionOption, ToolCall

from isaac.log_utils import log_context, log_event

logger = logging.getLogger(__name__)


class PermissionMixin:
    async def request_permission(
        self,
        options: list[PermissionOption],
        session_id: str,
        tool_call: ToolCall,
        **_: Any,
    ) -> RequestPermissionResponse | None:
        """Return a permission outcome, following Prompt Turn guidance for gated tools."""
        if self._conn is None:
            raise RuntimeError("Connection not established")
        requester = getattr(self._conn, "request_permission", None)
        if requester is None:
            raise RuntimeError("Connection missing request_permission handler")
        return await requester(options=options, session_id=session_id, tool_call=tool_call)

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
            with log_context(session_id=session_id, tool_call_id=tool_call_id):
                log_event(logger, "acp.permission.request", command=command.strip(), cwd=cwd or "")
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
                with log_context(session_id=session_id, tool_call_id=tool_call_id):
                    log_event(logger, "acp.permission.granted", mode="allow_always")
                return True
            if option_id == "allow_once":
                with log_context(session_id=session_id, tool_call_id=tool_call_id):
                    log_event(logger, "acp.permission.granted", mode="allow_once")
                return True
            if key in self._session_allowed_commands.get(session_id, set()):
                return True
            with log_context(session_id=session_id, tool_call_id=tool_call_id):
                log_event(logger, "acp.permission.denied")
            return False
        except Exception as exc:  # pragma: no cover - defensive fallback
            with log_context(session_id=session_id, tool_call_id=tool_call_id):
                log_event(logger, "acp.permission.error", level=logging.WARNING, error=str(exc))
            return False
