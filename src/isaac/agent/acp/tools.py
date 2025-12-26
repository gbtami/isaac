"""Tool call handlers for ACP prompt turns."""

from __future__ import annotations

import uuid
from typing import Any

from isaac.agent.tool_execution import ToolExecutionContext, execute_run_command_with_terminal, execute_tool


class ToolCallsMixin:
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

    def _tool_execution_context(self) -> ToolExecutionContext:
        return ToolExecutionContext(
            send_update=self._send_update,
            request_run_permission=self._request_run_permission,
            session_cwds=self._session_cwds,
            session_modes=self._session_modes,
            terminals=self._terminals,
            cancel_events=self._cancel_events,
            tool_output_limit=self._tool_output_limit,
            terminal_output_limit=self._terminal_output_limit,
            command_timeout_s=self._command_timeout_s,
        )

    async def _execute_tool(
        self,
        session_id: str,
        *,
        tool_name: str,
        tool_call_id: str | None = None,
        arguments: dict[str, Any] | None = None,
    ) -> None:
        await execute_tool(
            self._tool_execution_context(),
            session_id,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            arguments=arguments,
        )

    async def _execute_run_command_with_terminal(
        self,
        session_id: str,
        *,
        tool_call_id: str,
        arguments: dict[str, Any],
    ) -> None:
        await execute_run_command_with_terminal(
            self._tool_execution_context(),
            session_id,
            tool_call_id=tool_call_id,
            arguments=arguments,
        )
