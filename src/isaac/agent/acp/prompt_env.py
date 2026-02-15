"""ACP adapter for the prompt runner environment."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from acp.contrib.tool_calls import ToolCallTracker
from acp.helpers import (
    session_notification,
    text_block,
    tool_content,
    tool_diff_content,
    update_agent_message,
    update_agent_thought,
)

from isaac.agent.acp.plan_updates import build_plan_update
from isaac.agent.brain.events import ToolCallFinish, ToolCallStart
from isaac.agent.brain.tool_events import tool_kind
from isaac.agent.brain.prompt_runner import PromptEnv


@dataclass
class ACPPromptEnvAdapter:
    """Translate brain prompt events into ACP updates."""

    send_update: Callable[[Any], Awaitable[None]]

    def __post_init__(self) -> None:
        self._tool_trackers: dict[str, ToolCallTracker] = {}

    async def send_message_chunk(self, session_id: str, chunk: str) -> None:
        await self.send_update(session_notification(session_id, update_agent_message(text_block(chunk))))

    async def send_thought_chunk(self, session_id: str, chunk: str) -> None:
        await self.send_update(session_notification(session_id, update_agent_thought(text_block(chunk))))

    async def send_notification(self, session_id: str, message: str) -> None:
        await self.send_update(session_notification(session_id, update_agent_message(text_block(message))))

    async def send_tool_start(self, session_id: str, event: ToolCallStart) -> None:
        tracker = ToolCallTracker(id_factory=lambda: event.tool_call_id)
        self._tool_trackers[event.tool_call_id] = tracker
        start = tracker.start(
            external_id=event.tool_call_id,
            title=event.tool_name,
            kind=event.kind,
            status="in_progress",
            raw_input=event.raw_input,
        )
        await self.send_update(session_notification(session_id, start))

    async def send_tool_finish(self, session_id: str, event: ToolCallFinish) -> None:
        tracker = self._tool_trackers.pop(event.tool_call_id, None)
        if tracker is None:
            tracker = ToolCallTracker(id_factory=lambda: event.tool_call_id)
            start = tracker.start(
                external_id=event.tool_call_id,
                title=event.tool_name or "tool",
                kind=tool_kind(event.tool_name),
                status="in_progress",
                raw_input={"tool": event.tool_name},
            )
            await self.send_update(session_notification(session_id, start))

        summary = event.raw_output.get("error") or event.raw_output.get("content") or ""
        content_blocks: list[Any] = []
        if event.tool_name in {"edit_file", "apply_patch"} and isinstance(event.new_text, str):
            path = str(event.raw_output.get("path") or "")
            with contextlib.suppress(Exception):
                content_blocks.append(tool_diff_content(path, event.new_text, event.old_text))
        if not content_blocks and summary:
            content_blocks = [tool_content(text_block(str(summary)))]
        progress = tracker.progress(
            external_id=event.tool_call_id,
            status=event.status,
            raw_output=event.raw_output,
            content=content_blocks or None,
        )
        await self.send_update(session_notification(session_id, progress))

    async def send_plan_update(
        self,
        session_id: str,
        plan_steps: Any,
        active_index: int | None,
        status_all: str | None,
    ) -> None:
        update = build_plan_update(plan_steps, active_index=active_index, status_all=status_all)
        if update is None:
            return
        await self.send_update(session_notification(session_id, update))


def build_prompt_env(
    *,
    session_modes: dict[str, str],
    session_cwds: dict[str, Path],
    session_last_chunk: dict[str, str | None],
    send_update: Callable[[Any], Awaitable[None]],
    request_run_permission: Callable[[str, str, str, str | None], Awaitable[bool]],
    set_usage: Callable[[str, Any | None], None],
) -> PromptEnv:
    """Construct a PromptEnv wired to ACP update helpers."""

    adapter = ACPPromptEnvAdapter(send_update=send_update)
    return PromptEnv(
        session_modes=session_modes,
        session_cwds=session_cwds,
        session_last_chunk=session_last_chunk,
        send_message_chunk=adapter.send_message_chunk,
        send_thought_chunk=adapter.send_thought_chunk,
        send_tool_start=adapter.send_tool_start,
        send_tool_finish=adapter.send_tool_finish,
        send_plan_update=adapter.send_plan_update,
        send_notification=adapter.send_notification,
        send_protocol_update=send_update,
        request_run_permission=request_run_permission,
        set_usage=set_usage,
    )


__all__ = ["ACPPromptEnvAdapter", "build_prompt_env"]
