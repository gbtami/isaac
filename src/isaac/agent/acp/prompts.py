"""Prompt turn handlers for ACP."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from acp import PromptResponse
from acp.helpers import ContentBlock

from isaac.agent.brain.prompt_runner import PromptEnv
from isaac.agent.brain.prompt_handler import PromptHandler
from isaac.agent.plan_shortcuts import build_plan_shortcut_notification, parse_plan_shortcut
from isaac.agent.prompt_utils import extract_prompt_text
from isaac.agent.slash import handle_slash_command
from isaac.agent.tools import register_tools
from isaac.log_utils import log_context, log_event

logger = logging.getLogger(__name__)


class PromptMixin:
    def _build_prompt_handler(self) -> PromptHandler:
        """Construct the prompt handler."""
        env = PromptEnv(
            session_modes=self._session_modes,
            session_last_chunk=self._session_last_chunk,
            send_update=self._send_update,
            request_run_permission=lambda session_id, tool_call_id, command, cwd: self._request_run_permission(  # type: ignore[arg-type]
                session_id=session_id,
                tool_call_id=tool_call_id,
                command=command,
                cwd=cwd,
            ),
            set_usage=lambda session_id, usage: self._session_usage.__setitem__(session_id, usage),
        )
        return PromptHandler(
            env=env,
            register_tools=register_tools,
        )

    async def prompt(
        self,
        prompt: list[ContentBlock],
        session_id: str,
        **_: Any,
    ) -> PromptResponse:
        """Process a prompt turn per Prompt Turn lifecycle (session/prompt)."""
        with log_context(session_id=session_id):
            log_event(logger, "acp.prompt.request")
        cancel_event = self._cancel_events.setdefault(session_id, asyncio.Event())
        cancel_event.clear()
        # Reset last text chunk tracking for this prompt turn.
        self._session_last_chunk[session_id] = None

        for block in prompt:
            tool_call = getattr(block, "tool_call", None)
            if tool_call:
                await self._handle_tool_call(session_id, tool_call)
                return PromptResponse(stop_reason="end_turn")

        prompt_text = extract_prompt_text(prompt)
        # Persist the user prompt after capturing history so the current turn is not duplicated.
        self._store_user_prompt(session_id, prompt)

        slash = await handle_slash_command(self, session_id, prompt_text)
        if slash:
            await self._send_update(slash)
            return PromptResponse(stop_reason="end_turn")

        plan_request = parse_plan_shortcut(prompt_text)
        if plan_request:
            await self._send_update(build_plan_shortcut_notification(session_id, plan_request))
            return PromptResponse(stop_reason="end_turn")

        if cancel_event.is_set():
            return PromptResponse(stop_reason="cancelled")

        return await self._prompt_handler.handle_prompt(
            session_id,
            prompt_text,
            cancel_event,
        )

    async def cancel(self, session_id: str, **_: Any) -> None:
        """Stop in-flight prompt/tool work for a session (Prompt Turn cancellation)."""
        with log_context(session_id=session_id):
            log_event(logger, "acp.prompt.cancel")
        event = self._cancel_events.get(session_id)
        if event:
            event.set()
