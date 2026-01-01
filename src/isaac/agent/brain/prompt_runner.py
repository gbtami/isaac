"""Shared prompt handling helpers for planning/execution flows."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import logging
from typing import Any, Awaitable, Callable, Dict

from pydantic_ai.messages import FunctionToolCallEvent, FunctionToolResultEvent, RetryPromptPart
from isaac.agent.history_types import ChatMessage
from isaac.agent.brain.events import ToolCallFinish, ToolCallStart
from isaac.agent.brain.tool_events import should_record_tool_history, tool_history_summary, tool_kind
from isaac.agent.brain.tool_args import coerce_tool_args
from isaac.log_utils import log_event
from isaac.agent.tools.run_command import (
    RunCommandContext,
    pop_run_command_permission,
    reset_run_command_context,
    set_run_command_context,
    set_run_command_permission,
)

logger = logging.getLogger(__name__)


@dataclass
class PromptEnv:
    """Environment/state shared by the prompt runner."""

    session_modes: Dict[str, str]
    session_last_chunk: Dict[str, str | None]
    send_message_chunk: Callable[[str, str], Awaitable[None]]
    send_thought_chunk: Callable[[str, str], Awaitable[None]]
    send_tool_start: Callable[[str, ToolCallStart], Awaitable[None]]
    send_tool_finish: Callable[[str, ToolCallFinish], Awaitable[None]]
    send_plan_update: Callable[[str, Any, int | None, str | None], Awaitable[None]]
    send_notification: Callable[[str, str], Awaitable[None]]
    send_protocol_update: Callable[[Any], Awaitable[None]]
    request_run_permission: Callable[[str, str, str, str | None], Awaitable[bool]]
    set_usage: Callable[[str, Any | None], None]


class PromptRunner:
    """Utilities shared by prompt handling."""

    def __init__(self, env: PromptEnv) -> None:
        self.env = env

    def _make_chunk_sender(self, session_id: str) -> Callable[[str], Awaitable[None]]:
        async def _push_chunk(chunk: str) -> None:
            last = self.env.session_last_chunk.get(session_id)
            if chunk == last:
                return
            self.env.session_last_chunk[session_id] = chunk
            await self.env.send_message_chunk(session_id, chunk)

        return _push_chunk

    def _make_thought_sender(self, session_id: str) -> Callable[[str], Awaitable[None]]:
        async def _push_thought(chunk: str) -> None:
            await self.env.send_thought_chunk(session_id, chunk)

        return _push_thought

    def _build_runner_event_handler(
        self,
        session_id: str,
        run_command_ctx_tokens: Dict[str, Any],
        plan_progress: dict[str, Any] | None = None,
        record_history: Callable[[ChatMessage], None] | None = None,
    ) -> Callable[[Any], Awaitable[bool]]:
        tool_call_inputs: Dict[str, Dict[str, Any]] = {}

        async def _handle_runner_event(event: Any) -> bool:
            if isinstance(event, FunctionToolCallEvent):
                tool_name = getattr(event.part, "tool_name", None) or ""
                raw_args = getattr(event.part, "args", None)
                args = coerce_tool_args(raw_args)
                log_event(
                    logger,
                    "tool.call.requested",
                    level=logging.DEBUG,
                    tool=tool_name,
                    tool_call_id=event.tool_call_id,
                    args=args,
                )
                start = ToolCallStart(
                    tool_call_id=event.tool_call_id,
                    tool_name=tool_name,
                    kind=tool_kind(tool_name),
                    raw_input={"tool": tool_name, **args},
                )
                await self.env.send_tool_start(session_id, start)
                tool_call_inputs[event.tool_call_id] = args
                if tool_name == "run_command":
                    allowed = True
                    mode = self.env.session_modes.get(session_id, "ask")
                    if mode == "ask":
                        allowed = await self.env.request_run_permission(
                            session_id=session_id,
                            tool_call_id=event.tool_call_id,
                            command=str(args.get("command") or ""),
                            cwd=args.get("cwd"),
                        )
                    set_run_command_permission(event.tool_call_id, allowed)

                    async def _permission(_: str, __: str | None = None) -> bool:
                        return allowed

                    token = set_run_command_context(RunCommandContext(request_permission=_permission))
                    run_command_ctx_tokens[event.tool_call_id] = token
                    if not allowed:
                        denied = ToolCallFinish(
                            tool_call_id=event.tool_call_id,
                            tool_name=tool_name,
                            status="failed",
                            raw_output={
                                "tool": tool_name,
                                "content": None,
                                "error": "permission denied",
                            },
                        )
                        await self.env.send_tool_finish(session_id, denied)
                        return True
                return True

            if isinstance(event, FunctionToolResultEvent):
                token = run_command_ctx_tokens.pop(event.tool_call_id, None)
                if token is not None:
                    reset_run_command_context(token)
                call_input = tool_call_inputs.pop(event.tool_call_id, {})
                result_part = event.result
                tool_name = getattr(result_part, "tool_name", None) or ""
                if tool_name == "run_command":
                    pop_run_command_permission(event.tool_call_id)
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
                new_text = raw_output.get("new_text")
                old_text = raw_output.get("old_text")
                finish = ToolCallFinish(
                    tool_call_id=event.tool_call_id,
                    tool_name=tool_name,
                    status=status,
                    raw_output=raw_output,
                    old_text=old_text if isinstance(old_text, str) else None,
                    new_text=new_text if isinstance(new_text, str) else None,
                )
                await self.env.send_tool_finish(session_id, finish)

                if record_history and should_record_tool_history(tool_name):
                    summary = tool_history_summary(tool_name, raw_output, status, raw_input=call_input)
                    if summary:
                        with contextlib.suppress(Exception):
                            record_history({"role": "assistant", "content": summary})

                if plan_progress and plan_progress.get("plan") and not raw_output.get("error"):
                    entries = getattr(plan_progress["plan"], "entries", []) or []
                    if entries:
                        idx = plan_progress.get("idx", 0)
                        idx = max(int(idx), 0)
                        if idx < len(entries):
                            idx += 1
                            plan_progress["idx"] = idx
                            if idx >= len(entries):
                                await self.env.send_plan_update(session_id, plan_progress["plan"], None, "completed")
                            else:
                                await self.env.send_plan_update(session_id, plan_progress["plan"], idx, None)
                return True

            return False

        return _handle_runner_event

    @staticmethod
    def _prompt_cancel() -> Any:
        from isaac.agent.brain.prompt_result import PromptResult

        return PromptResult(stop_reason="cancelled")

    @staticmethod
    def _prompt_end() -> Any:
        from isaac.agent.brain.prompt_result import PromptResult

        return PromptResult(stop_reason="end_turn")

    # Tool history helpers moved to isaac.agent.brain.tool_events.
