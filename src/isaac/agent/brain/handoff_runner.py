"""Handoff-specific planning/execution runner."""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict
from uuid import uuid4

from acp.helpers import (
    plan_entry,
    session_notification,
    text_block,
    update_agent_message,
    update_plan,
)
from acp.schema import PlanEntry
from pydantic_ai.run import AgentRunResultEvent  # type: ignore

from isaac.agent.brain.planner import parse_plan_from_text
from isaac.agent.brain.strategy_plan import PlanStep, PlanSteps
from isaac.agent.brain.strategy_runner import StrategyPromptRunner
from isaac.agent.brain.strategy_utils import plan_with_status, prepare_executor_prompt
from isaac.agent.runner import stream_with_runner
from isaac.agent.usage import normalize_usage


class HandoffRunner(StrategyPromptRunner):
    """Handles planning then execution for the handoff strategy."""

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
            planned = plan_with_status(plan_update, active_index=0)
            await self.env.send_update(session_notification(session_id, planned))

        executor_prompt = prepare_executor_prompt(
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
            completed = plan_with_status(plan_update, status_all="completed")
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
        tool_trackers: Dict[str, Any] = {}
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
            initial = plan_with_status(exec_plan_update, active_index=0)
            await self.env.send_update(session_notification(session_id, initial))
        if response_text == "":
            await _push_chunk(response_text)
        combined_usage = normalize_usage(usage) or normalize_usage(plan_usage)
        self.env.set_usage(session_id, combined_usage)
        return self._prompt_end()
