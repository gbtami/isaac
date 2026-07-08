"""Permission handling for ACP requests."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from acp import RequestPermissionResponse
from acp.helpers import text_block, tool_content
from acp.schema import PermissionOption, ToolCall

from isaac.agent.tools.policy import (
    approval_always_option_id,
    permission_body,
    permission_cache_key,
    permission_title,
    tool_permission_kind,
)
from isaac.log_utils import log_context, log_event

logger = logging.getLogger(__name__)


class PermissionMixin:
    async def request_permission(
        self,
        session_id: str,
        tool_call: ToolCall,
        options: list[PermissionOption],
        **_: Any,
    ) -> RequestPermissionResponse | None:
        """Return a permission outcome, following Prompt Turn guidance for gated tools."""
        if self._conn is None:
            raise RuntimeError("Connection not established")
        requester = getattr(self._conn, "request_permission", None)
        if requester is None:
            raise RuntimeError("Connection missing request_permission handler")
        return await requester(session_id=session_id, tool_call=tool_call, options=options)

    async def _request_tool_permission(
        self,
        session_id: str,
        *,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> bool:
        """Ask the client for permission to run a risky tool call.

        The allow-always cache is scoped by tool-specific stable keys. For
        ``run_command`` this keeps Isaac's historical command+cwd caching
        behaviour, while writes/network/delegates use path/host/task keys.
        """

        key = permission_cache_key(tool_name, arguments)
        if key in self._session_allowed_commands.get(session_id, set()):
            return True

        try:
            with log_context(session_id=session_id, tool_call_id=tool_call_id):
                log_event(
                    logger,
                    "acp.permission.request",
                    tool=tool_name,
                    permission_key=key,
                )
            allow_always_id = approval_always_option_id(tool_name)
            options = [
                PermissionOption(option_id="allow_once", name="Allow once", kind="allow_once"),
                PermissionOption(option_id=allow_always_id, name="Allow matching calls", kind="allow_always"),
                PermissionOption(option_id="reject_once", name="Reject", kind="reject_once"),
            ]
            cwd_display = str(self._session_cwds.get(session_id, Path.cwd()))
            title = permission_title(tool_name, arguments)
            body = permission_body(tool_name, arguments, cwd_display=cwd_display)
            tool_call = ToolCall(
                tool_call_id=tool_call_id,
                title=title,
                kind=tool_permission_kind(tool_name),
                raw_input={"tool": tool_name, **arguments},
                content=[tool_content(text_block(body))],
                status="pending",
            )
            from acp.schema import ToolCallUpdate  # type: ignore

            tool_call_update = ToolCallUpdate.model_validate(
                getattr(tool_call, "model_dump", lambda **_: tool_call)(by_alias=True)
            )
            requester = getattr(self._conn, "request_permission", None)
            if requester is None:
                raise RuntimeError("Connection missing request_permission handler")
            resp = await requester(session_id=session_id, tool_call=tool_call_update, options=options)
            outcome = getattr(resp, "outcome", None)
            option_id = getattr(outcome, "option_id", "") if outcome is not None else ""
            if option_id in {allow_always_id, "allow_always"}:
                self._session_allowed_commands.setdefault(session_id, set()).add(key)
                with log_context(session_id=session_id, tool_call_id=tool_call_id):
                    log_event(logger, "acp.permission.granted", tool=tool_name, mode="allow_always")
                return True
            if option_id == "allow_once":
                with log_context(session_id=session_id, tool_call_id=tool_call_id):
                    log_event(logger, "acp.permission.granted", tool=tool_name, mode="allow_once")
                return True
            if key in self._session_allowed_commands.get(session_id, set()):
                return True
            with log_context(session_id=session_id, tool_call_id=tool_call_id):
                log_event(logger, "acp.permission.denied", tool=tool_name)
            return False
        except Exception as exc:  # pragma: no cover - defensive fallback
            with log_context(session_id=session_id, tool_call_id=tool_call_id):
                log_event(logger, "acp.permission.error", level=logging.WARNING, tool=tool_name, error=str(exc))
            return False

    async def _request_run_permission(
        self,
        session_id: str,
        *,
        tool_call_id: str,
        command: str,
        cwd: str | None,
    ) -> bool:
        """Ask the client for permission to run a shell command (ACP permission flow)."""

        return await self._request_tool_permission(
            session_id,
            tool_call_id=tool_call_id,
            tool_name="run_command",
            arguments={"command": command, "cwd": cwd},
        )
