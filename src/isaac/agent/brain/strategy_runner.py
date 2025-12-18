"""Shared prompt handling helpers for planning/execution flows."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict

from acp.contrib.tool_calls import ToolCallTracker
from acp.helpers import (
    session_notification,
    text_block,
    tool_content,
    tool_diff_content,
    update_agent_message,
    update_agent_thought,
)
from acp.schema import SessionNotification
from pydantic_ai.messages import FunctionToolCallEvent, FunctionToolResultEvent, RetryPromptPart
from isaac.agent.brain.strategy_utils import coerce_tool_args, plan_with_status
from isaac.agent.tools.run_command import (
    RunCommandContext,
    pop_run_command_permission,
    reset_run_command_context,
    set_run_command_context,
    set_run_command_permission,
)


@dataclass
class StrategyEnv:
    """Environment/state shared by the strategy runner."""

    session_modes: Dict[str, str]
    session_last_chunk: Dict[str, str | None]
    send_update: Callable[[SessionNotification], Awaitable[None]]
    request_run_permission: Callable[[str, str, str, str | None], Awaitable[bool]]
    set_usage: Callable[[str, Any | None], None]


class StrategyPromptRunner:
    """Utilities shared by prompt strategies."""

    def __init__(self, env: StrategyEnv) -> None:
        self.env = env

    def _make_chunk_sender(self, session_id: str) -> Callable[[str], Awaitable[None]]:
        async def _push_chunk(chunk: str) -> None:
            last = self.env.session_last_chunk.get(session_id)
            if chunk == last:
                return
            self.env.session_last_chunk[session_id] = chunk
            await self.env.send_update(
                session_notification(
                    session_id,
                    update_agent_message(text_block(chunk)),
                )
            )

        return _push_chunk

    def _make_thought_sender(self, session_id: str) -> Callable[[str], Awaitable[None]]:
        async def _push_thought(chunk: str) -> None:
            await self.env.send_update(
                session_notification(
                    session_id,
                    update_agent_thought(text_block(chunk)),
                )
            )

        return _push_thought

    def _build_runner_event_handler(
        self,
        session_id: str,
        tool_trackers: Dict[str, ToolCallTracker],
        run_command_ctx_tokens: Dict[str, Any],
        plan_progress: dict[str, Any] | None = None,
        record_history: Callable[[dict[str, str]], None] | None = None,
    ) -> Callable[[Any], Awaitable[bool]]:
        async def _handle_runner_event(event: Any) -> bool:
            if isinstance(event, FunctionToolCallEvent):
                tool_name = getattr(event.part, "tool_name", None) or ""
                raw_args = getattr(event.part, "args", None)
                args = coerce_tool_args(raw_args)
                tracker = ToolCallTracker(id_factory=lambda: event.tool_call_id)
                tool_trackers[event.tool_call_id] = tracker
                start = tracker.start(
                    external_id=event.tool_call_id,
                    title=tool_name,
                    status="in_progress",
                    raw_input={"tool": tool_name, **args},
                )
                await self.env.send_update(session_notification(session_id, start))
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
                        denied = tracker.progress(
                            external_id=event.tool_call_id,
                            status="failed",
                            raw_output={
                                "tool": tool_name,
                                "content": None,
                                "error": "permission denied",
                            },
                            content=[tool_content(text_block("Command blocked: permission denied"))],
                        )
                        await self.env.send_update(session_notification(session_id, denied))
                        return True
                return True

            if isinstance(event, FunctionToolResultEvent):
                token = run_command_ctx_tokens.pop(event.tool_call_id, None)
                if token is not None:
                    reset_run_command_context(token)
                tracker = tool_trackers.pop(event.tool_call_id, None) or ToolCallTracker(
                    id_factory=lambda: event.tool_call_id
                )
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
                summary = raw_output.get("error") or raw_output.get("content") or ""
                content_blocks: list[Any] = []
                new_text = raw_output.get("new_text")
                old_text = raw_output.get("old_text")
                if tool_name in {"edit_file", "apply_patch"} and isinstance(new_text, str):
                    path = str(raw_output.get("path") or "")
                    with contextlib.suppress(Exception):
                        content_blocks.append(tool_diff_content(path, new_text, old_text))
                if not content_blocks and summary:
                    content_blocks = [tool_content(text_block(str(summary)))]
                progress = tracker.progress(
                    external_id=event.tool_call_id,
                    status=status,
                    raw_output=raw_output,
                    content=content_blocks or None,
                )
                await self.env.send_update(session_notification(session_id, progress))

                if record_history:
                    summary = self._tool_history_summary(tool_name, raw_output, status)
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
                                plan_note = plan_with_status(plan_progress["plan"], status_all="completed")
                            else:
                                plan_note = plan_with_status(plan_progress["plan"], active_index=idx)
                            await self.env.send_update(session_notification(session_id, plan_note))
                return True

            return False

        return _handle_runner_event

    @staticmethod
    def _prompt_cancel() -> Any:
        from acp.schema import PromptResponse  # local import to avoid cycles

        return PromptResponse(stop_reason="cancelled")

    @staticmethod
    def _prompt_end() -> Any:
        from acp.schema import PromptResponse  # local import to avoid cycles

        return PromptResponse(stop_reason="end_turn")

    @staticmethod
    def _tool_history_summary(tool_name: str, raw_output: dict[str, Any], status: str) -> str | None:
        content = raw_output.get("content")
        summary_prefix = f"{tool_name} ({status})"
        if tool_name == "run_command":
            cmd = raw_output.get("command") or raw_output.get("cmd")
            cwd = raw_output.get("cwd")
            cmd_str = str(cmd).strip() if cmd else ""
            cwd_str = f" (cwd: {cwd})" if cwd else ""
            if cmd_str:
                return f"Ran command: {cmd_str}{cwd_str} [{status}]"
            return f"Ran command [{status}]"
        if tool_name in {"edit_file", "apply_patch"}:
            path = raw_output.get("path")
            path_str = f" {path}" if path else ""
            return f"Updated file{path_str} [{status}]"
        if tool_name == "read_file":
            path = raw_output.get("path")
            if path:
                return f"Read file {path} [{status}]"
        if tool_name == "list_files":
            root = raw_output.get("directory") or raw_output.get("path")
            if root:
                return f"Listed files in {root} [{status}]"
        if tool_name == "file_summary":
            path = raw_output.get("path")
            if path:
                return f"Summarized file {path} [{status}]"
        if tool_name == "code_search":
            pattern = raw_output.get("pattern")
            directory = raw_output.get("directory")
            if pattern:
                where = f" in {directory}" if directory else ""
                return f"Searched for '{pattern}'{where} [{status}]"
        if tool_name == "fetch_url":
            fetched = raw_output.get("url") or raw_output.get("source") or raw_output.get("request_url")
            status_code = raw_output.get("status_code")
            detail = f" ({status_code})" if status_code else ""
            if fetched:
                return f"Fetched URL {fetched}{detail} [{status}]"
        if tool_name:
            return f"{summary_prefix}"
        if content:
            return f"Tool result [{status}]: {content}"
        return None
