"""Prompt handling helpers for the hand-off planning/execution flow."""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Literal
from uuid import uuid4

from acp.contrib.tool_calls import ToolCallTracker
from acp.helpers import (
    plan_entry,
    session_notification,
    text_block,
    tool_content,
    tool_diff_content,
    update_agent_message,
    update_agent_thought,
    update_plan,
)
from acp.schema import PlanEntry, SessionNotification
from pydantic import BaseModel, model_validator
from pydantic_ai import Agent as PydanticAgent  # type: ignore
from pydantic_ai.messages import FunctionToolCallEvent, FunctionToolResultEvent, RetryPromptPart
from pydantic_ai.run import AgentRunResultEvent  # type: ignore

from isaac.agent.brain.planner import parse_plan_from_text
from isaac.agent.brain.prompt import EXECUTOR_INSTRUCTIONS, PLANNER_INSTRUCTIONS, SYSTEM_PROMPT
from isaac.agent.runner import stream_with_runner
from isaac.agent.tools.run_command import (
    RunCommandContext,
    pop_run_command_permission,
    reset_run_command_context,
    set_run_command_context,
    set_run_command_permission,
)
from isaac.agent.usage import normalize_usage


class PlanStep(BaseModel):
    content: str
    priority: Literal["high", "medium", "low"] = "medium"
    id: str | None = None


class PlanSteps(BaseModel):
    entries: List[PlanStep]

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_formats(cls, value: Any) -> Any:
        """Accept older/looser plan shapes and normalize to entries."""
        entries: Any = None
        if isinstance(value, dict):
            if "entries" in value:
                entries = value.get("entries")
            elif "steps" in value:
                entries = value.get("steps")
        else:
            entries = value

        if isinstance(entries, list):
            normalized: list[dict[str, Any]] = []
            for item in entries:
                if isinstance(item, PlanStep):
                    normalized.append(item.model_dump())
                elif isinstance(item, str):
                    text = item.strip()
                    if text:
                        normalized.append({"content": text})
                elif isinstance(item, dict):
                    if "content" not in item and "step" in item:
                        item = {**item, "content": item.get("step")}
                    normalized.append(item)
            return {"entries": normalized}

        if isinstance(entries, str):
            lines = [ln.strip() for ln in entries.splitlines() if ln.strip()]
            if not lines:
                return {"entries": []}
            items: list[str] = []
            for ln in lines:
                cleaned = ln.lstrip("-* ").strip()
                if cleaned and cleaned[0].isdigit() and "." in cleaned:
                    cleaned = cleaned.split(".", 1)[1].strip()
                if cleaned:
                    items.append(cleaned)
            return {"entries": [{"content": it} for it in items]} if items else {"entries": []}

        return value


def build_planning_agent(model: Any, model_settings: Any = None) -> PydanticAgent:
    """Create a lightweight planning agent for programmatic hand-off."""

    return PydanticAgent(
        model,
        system_prompt=SYSTEM_PROMPT,
        instructions=PLANNER_INSTRUCTIONS,
        model_settings=model_settings,
        toolsets=(),
        output_type=PlanSteps,
    )


@dataclass
class HandoffEnv:
    """Environment/state shared by the hand-off runner."""

    session_modes: Dict[str, str]
    session_last_chunk: Dict[str, str | None]
    send_update: Callable[[SessionNotification], Awaitable[None]]
    request_run_permission: Callable[[str, str, str, str | None], Awaitable[bool]]
    set_usage: Callable[[str, Any | None], None]


class HandoffPromptRunner:
    """Handles planning then execution for a single prompt flow."""

    def __init__(self, env: HandoffEnv) -> None:
        self.env = env

    async def run(
        self,
        session_id: str,
        prompt_text: str,
        *,
        planner_history: list[Any],
        executor_history: list[Any],
        cancel_event: asyncio.Event,
        runner: Any,
        planner: Any,
        store_planner_messages: Callable[[Any], None],
        store_executor_messages: Callable[[Any], None],
    ) -> Any:
        plan_update, plan_text, plan_usage = await self._run_planning_phase(
            session_id,
            prompt_text,
            history=planner_history,
            cancel_event=cancel_event,
            planner=planner,
            store_model_messages=store_planner_messages,
        )
        if plan_update:
            planned = self._plan_with_status(plan_update, active_index=0)
            await self.env.send_update(session_notification(session_id, planned))

        executor_prompt = self._prepare_executor_prompt(
            prompt_text,
            plan_update=plan_update,
            plan_response=plan_text,
        )
        response = await self._run_execution_phase(
            session_id,
            runner,
            executor_prompt,
            history=executor_history,
            cancel_event=cancel_event,
            store_model_messages=store_executor_messages,
            plan_update=plan_update,
            plan_response=plan_text,
            plan_usage=plan_usage,
            log_context="executor",
        )
        if plan_update:
            completed = self._plan_with_status(plan_update, status_all="completed")
            await self.env.send_update(session_notification(session_id, completed))
        return response

    async def _run_planning_phase(
        self,
        session_id: str,
        prompt_text: str,
        *,
        history: Any,
        cancel_event: asyncio.Event,
        planner: Any,
        store_model_messages: Callable[[Any], None],
    ) -> tuple[Any | None, str | None, Any | None]:
        plan_chunks: list[str] = []
        structured_plan: PlanSteps | None = None

        async def _plan_chunk(chunk: str) -> None:
            plan_chunks.append(chunk)

        async def _capture_plan_event(event: Any) -> bool:
            nonlocal structured_plan
            if isinstance(event, AgentRunResultEvent):
                output = event.result.output
                if isinstance(output, PlanSteps):
                    structured_plan = output
            return False

        plan_response, plan_usage = await stream_with_runner(
            planner,
            prompt_text,
            _plan_chunk,
            self._make_thought_sender(session_id),
            cancel_event,
            history=history,
            on_event=_capture_plan_event,
            store_messages=store_model_messages,
            log_context="planner",
        )
        combined_plan_text = plan_response or "".join(plan_chunks)
        if combined_plan_text.startswith("Provider error:"):
            msg = combined_plan_text.removeprefix("Provider error:").strip()
            await self.env.send_update(
                session_notification(
                    session_id,
                    update_agent_message(text_block(f"Model/provider error during planning: {msg}")),
                )
            )
            combined_plan_text = ""
        elif combined_plan_text.lower().startswith("model output failed validation"):
            await self.env.send_update(
                session_notification(
                    session_id,
                    update_agent_message(text_block(combined_plan_text)),
                )
            )
            combined_plan_text = ""
        plan_update = None
        plan_text_for_executor = combined_plan_text or plan_response

        if structured_plan and structured_plan.entries:
            entries: list[PlanEntry] = []
            executor_lines: list[str] = []
            for idx, item in enumerate(structured_plan.entries):
                if not isinstance(item, PlanStep):
                    continue
                content = item.content.strip()
                if not content:
                    continue
                priority = item.priority if item.priority in {"high", "medium", "low"} else "medium"
                step_id = item.id or f"step_{idx + 1}_{uuid4().hex[:6]}"
                pe = plan_entry(content, priority=priority, status="pending")
                pe = pe.model_copy(update={"field_meta": {"id": step_id}})
                entries.append(pe)
                executor_lines.append(content)
            if entries:
                plan_update = update_plan(entries)
                plan_text_for_executor = "\n".join(f"- {line}" for line in executor_lines)

        if plan_update is None:
            plan_update = parse_plan_from_text(combined_plan_text or "")
            if not plan_update and combined_plan_text:
                entries = [plan_entry(combined_plan_text)]
                plan_update = update_plan(entries)

        return plan_update, plan_text_for_executor, plan_usage

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
    ) -> Callable[[Any], Awaitable[bool]]:
        async def _handle_runner_event(event: Any) -> bool:
            if isinstance(event, FunctionToolCallEvent):
                tool_name = getattr(event.part, "tool_name", None) or ""
                raw_args = getattr(event.part, "args", None)
                args = self._coerce_tool_args(raw_args)
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

                if plan_progress and plan_progress.get("plan") and not raw_output.get("error"):
                    entries = getattr(plan_progress["plan"], "entries", []) or []
                    if entries:
                        idx = plan_progress.get("idx", 0)
                        idx = max(int(idx), 0)
                        if idx < len(entries):
                            idx += 1
                            plan_progress["idx"] = idx
                            if idx >= len(entries):
                                plan_note = self._plan_with_status(plan_progress["plan"], status_all="completed")
                            else:
                                plan_note = self._plan_with_status(plan_progress["plan"], active_index=idx)
                            await self.env.send_update(session_notification(session_id, plan_note))
                return True

            return False

        return _handle_runner_event

    async def _run_execution_phase(
        self,
        session_id: str,
        runner: Any,
        executor_prompt: str,
        *,
        history: Any,
        cancel_event: asyncio.Event,
        store_model_messages: Callable[[Any], None],
        plan_update: Any | None = None,
        plan_response: str | None = None,
        plan_usage: Any | None = None,
        allow_plan_parse: bool = True,
        log_context: str | None = None,
    ) -> Any:
        tool_trackers: Dict[str, ToolCallTracker] = {}
        run_command_ctx_tokens: Dict[str, Any] = {}
        plan_progress = {"plan": plan_update, "idx": 0} if plan_update else None
        _push_chunk = self._make_chunk_sender(session_id)
        _push_thought = self._make_thought_sender(session_id)
        handler = self._build_runner_event_handler(
            session_id,
            tool_trackers,
            run_command_ctx_tokens,
            plan_progress,
        )

        response_text, usage = await stream_with_runner(
            runner,
            executor_prompt,
            _push_chunk,
            _push_thought,
            cancel_event,
            history=history,
            on_event=handler,
            store_messages=store_model_messages,
            log_context=log_context,
        )
        if response_text is None:
            return self._prompt_cancel()
        if response_text.startswith("Provider error:"):
            msg = response_text.removeprefix("Provider error:").strip()
            await self.env.send_update(
                session_notification(
                    session_id,
                    update_agent_message(text_block(f"Model/provider error: {msg}")),
                )
            )
            return self._prompt_end()

        exec_plan_update = None
        if allow_plan_parse and not plan_update and not plan_response:
            exec_plan_update = parse_plan_from_text(response_text or "")
        if exec_plan_update:
            plan_progress = {"plan": exec_plan_update, "idx": 0}
            initial = self._plan_with_status(exec_plan_update, active_index=0)
            await self.env.send_update(session_notification(session_id, initial))
        if response_text == "":
            await _push_chunk(response_text)
        combined_usage = normalize_usage(usage) or normalize_usage(plan_usage)
        self.env.set_usage(session_id, combined_usage)
        return self._prompt_end()

    @staticmethod
    def _prepare_executor_prompt(
        prompt_text: str,
        *,
        plan_update: Any | None = None,
        plan_response: str | None = None,
    ) -> str:
        plan_lines = [getattr(e, "content", "") for e in getattr(plan_update, "entries", []) or []]
        if plan_lines:
            plan_block = "\n".join(f"- {line}" for line in plan_lines if line)
            return f"{prompt_text}\n\nPlan:\n{plan_block}\n\n{EXECUTOR_INSTRUCTIONS}"
        if plan_response:
            return f"{prompt_text}\n\nPlan:\n{plan_response}\n\n{EXECUTOR_INSTRUCTIONS}"
        return prompt_text

    @staticmethod
    def _plan_with_status(plan_update: Any, *, active_index: int | None = None, status_all: str | None = None) -> Any:
        try:
            entries = list(getattr(plan_update, "entries", []) or [])
        except Exception:
            return plan_update
        if not entries:
            return plan_update
        updated: list[PlanEntry] = []
        for idx, entry in enumerate(entries):
            if status_all is not None:
                status = status_all
            elif active_index is not None:
                if idx < active_index:
                    status = "completed"
                elif idx == active_index:
                    status = "in_progress"
                else:
                    status = "pending"
            else:
                status = getattr(entry, "status", "pending")
            try:
                updated.append(entry.model_copy(update={"status": status}))
            except Exception:
                updated.append(entry)
        try:
            return update_plan(updated)
        except Exception:
            return plan_update

    @staticmethod
    def _prompt_cancel() -> Any:
        from acp.schema import PromptResponse  # local import to avoid cycles

        return PromptResponse(stop_reason="cancelled")

    @staticmethod
    def _prompt_end() -> Any:
        from acp.schema import PromptResponse  # local import to avoid cycles

        return PromptResponse(stop_reason="end_turn")

    @staticmethod
    def _coerce_tool_args(raw_args: Any) -> dict[str, Any]:
        """Convert tool call args to a dict, handling common non-dict shapes."""

        if raw_args is None:
            return {}
        if isinstance(raw_args, dict):
            return dict(raw_args)
        if isinstance(raw_args, str):
            return {"command": raw_args}

        for attr in ("model_dump", "dict"):
            func = getattr(raw_args, attr, None)
            if callable(func):
                try:
                    data = func()
                    if isinstance(data, dict):
                        return dict(data)
                except Exception:
                    continue

        collected: dict[str, Any] = {}
        for key in ("command", "cwd", "timeout"):
            if hasattr(raw_args, key):
                try:
                    collected[key] = getattr(raw_args, key)
                except Exception:
                    continue

        if collected:
            return collected

        try:
            mapping = dict(raw_args)  # type: ignore[arg-type]
            if isinstance(mapping, dict):
                return mapping
        except Exception:
            return {}
        return {}
