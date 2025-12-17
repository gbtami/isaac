"""Single-agent prompt strategy with an embedded todo planning tool."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict

from acp.helpers import session_notification, text_block, update_agent_message
from acp.schema import PromptResponse
from pydantic_ai import RunContext
from pydantic_ai.messages import FunctionToolResultEvent  # type: ignore

from isaac.agent import models as model_registry
from isaac.agent.brain.strategy_runner import StrategyEnv, StrategyPromptRunner
from isaac.agent.brain.planner import parse_plan_from_text
from isaac.agent.brain.strategy_base import ModelBuildError, PromptStrategy
from isaac.agent.brain.strategy_plan import PlanSteps
from isaac.agent.brain.strategy_utils import (
    create_subagent_for_model,
    create_subagent_planner_for_model,
    plan_update_from_steps,
    plan_with_status,
)
from isaac.agent.brain.prompt import TODO_PLANNER_INSTRUCTIONS
from isaac.agent.runner import register_tools as default_register_tools, stream_with_runner
from isaac.agent.tools import DEFAULT_TOOL_TIMEOUT_S
from isaac.agent.usage import normalize_usage


@dataclass
class SubagentSessionState:
    """State for sessions using the subagent planning strategy."""

    runner: Any | None = None
    todo_planner: Any | None = None
    model_id: str | None = None
    history: list[Any] = field(default_factory=list)
    model_error: str | None = None
    model_error_notified: bool = False


class SubagentPromptStrategy(PromptStrategy):
    """Single-runner strategy that delegates planning to a subagent todo tool."""

    def __init__(
        self,
        env: StrategyEnv,
        *,
        register_tools: Any | None = None,
    ) -> None:
        self.env = env
        self._register_tools = register_tools or default_register_tools
        self._handoff_helper = StrategyPromptRunner(env)
        self._sessions: Dict[str, SubagentSessionState] = {}

    async def init_session(self, session_id: str, toolsets: list[Any]) -> None:
        state = self._sessions.setdefault(
            session_id,
            SubagentSessionState(model_id=model_registry.current_model_id()),
        )
        await self._build_runner(session_id, state, toolsets)

    async def set_session_model(self, session_id: str, model_id: str, toolsets: list[Any]) -> None:
        state = self._sessions.setdefault(
            session_id,
            SubagentSessionState(model_id=model_registry.current_model_id()),
        )
        try:
            executor = create_subagent_for_model(model_id, self._register_tools, toolsets=toolsets)
            planner = create_subagent_planner_for_model(model_id)
            state.runner = executor
            state.todo_planner = planner
            self._attach_todo_tool(state)
            state.model_id = model_id
            state.history = []
            state.model_error = None
            state.model_error_notified = False
        except Exception as exc:
            msg = f"Model load failed: {exc}"
            state.runner = None
            state.todo_planner = None
            state.model_error = msg
            state.model_error_notified = False
            await self.env.send_update(
                session_notification(
                    session_id,
                    update_agent_message(text_block(msg)),
                )
            )
            raise ModelBuildError(msg) from exc

    async def handle_prompt(
        self,
        session_id: str,
        prompt_text: str,
        cancel_event: asyncio.Event,
    ) -> PromptResponse:
        state = self._sessions.setdefault(
            session_id,
            SubagentSessionState(model_id=model_registry.current_model_id()),
        )
        if state.runner is None:
            await self._build_runner(session_id, state, toolsets=None)
        if state.model_error or state.runner is None:
            return await self._respond_model_error(session_id, state)

        history = list(state.history)
        plan_progress: dict[str, Any] | None = {"plan": None, "idx": 0}
        _push_chunk = self._handoff_helper._make_chunk_sender(session_id)  # type: ignore[attr-defined]
        _push_thought = self._handoff_helper._make_thought_sender(session_id)  # type: ignore[attr-defined]

        tool_trackers: Dict[str, Any] = {}
        run_command_ctx_tokens: Dict[str, Any] = {}

        async def _maybe_capture_plan(event: Any) -> None:
            if not isinstance(event, FunctionToolResultEvent):
                return
            result_part = event.result
            tool_name = getattr(result_part, "tool_name", None) or ""
            if tool_name != "todo":
                return
            plan_update = self._plan_from_todo_result(result_part)
            if plan_update is None:
                return
            if plan_progress is not None:
                plan_progress["plan"] = plan_update
                plan_progress["idx"] = 0
            initial = plan_with_status(plan_update, active_index=0)
            await self.env.send_update(session_notification(session_id, initial))

        base_handler = self._handoff_helper._build_runner_event_handler(  # type: ignore[attr-defined]
            session_id,
            tool_trackers,
            run_command_ctx_tokens,
            plan_progress,
        )

        async def _on_event(event: Any) -> bool:
            is_todo_result = (
                isinstance(event, FunctionToolResultEvent)
                and getattr(getattr(event, "result", None), "tool_name", "") == "todo"
            )
            if is_todo_result and plan_progress is not None:
                saved_plan = plan_progress.get("plan")
                saved_idx = plan_progress.get("idx", 0)
                plan_progress["plan"] = None
                plan_progress["idx"] = saved_idx
                handled = await base_handler(event)
                plan_progress["plan"] = saved_plan
                plan_progress["idx"] = saved_idx
                await _maybe_capture_plan(event)
                return handled
            await _maybe_capture_plan(event)
            return await base_handler(event)

        def _store_messages(msgs: Any) -> None:
            try:
                state.history.extend(list(msgs or []))
            except Exception:
                return

        response_text, usage = await stream_with_runner(
            state.runner,
            prompt_text,
            _push_chunk,
            _push_thought,
            cancel_event,
            history=history,
            on_event=_on_event,
            store_messages=_store_messages,
            log_context="subagent",
        )
        if response_text is None:
            return self._handoff_helper._prompt_cancel()  # type: ignore[attr-defined]
        if response_text.startswith("Provider error:"):
            msg = response_text.removeprefix("Provider error:").strip()
            await self.env.send_update(
                session_notification(
                    session_id,
                    update_agent_message(text_block(f"Model/provider error: {msg}")),
                )
            )
            return self._handoff_helper._prompt_end()  # type: ignore[attr-defined]
        if plan_progress and plan_progress.get("plan"):
            completed = plan_with_status(plan_progress["plan"], status_all="completed")
            await self.env.send_update(session_notification(session_id, completed))
        self.env.set_usage(session_id, normalize_usage(usage))
        return self._handoff_helper._prompt_end()  # type: ignore[attr-defined]

    def model_id(self, session_id: str) -> str | None:
        return self._sessions.get(session_id, SubagentSessionState()).model_id

    def set_session_runner(self, session_id: str, runner: Any | None) -> None:
        state = self._sessions.setdefault(
            session_id,
            SubagentSessionState(model_id=model_registry.current_model_id()),
        )
        state.runner = runner
        try:
            state.todo_planner = create_subagent_planner_for_model(state.model_id or model_registry.current_model_id())
        except Exception:
            state.todo_planner = None
        state.model_id = state.model_id or model_registry.current_model_id()
        state.model_error = None
        state.model_error_notified = False
        self._attach_todo_tool(state)

    def session_ids(self) -> list[str]:
        return list(self._sessions.keys())

    def _attach_todo_tool(self, state: SubagentSessionState) -> None:
        runner = state.runner
        planner = state.todo_planner
        if runner is None:
            return
        if not hasattr(runner, "tool"):
            return
        if getattr(runner, "_isaac_todo_registered", False):
            return

        @runner.tool(name="todo", timeout=DEFAULT_TOOL_TIMEOUT_S)  # type: ignore[misc]
        async def todo_tool(ctx: RunContext[Any], task: str) -> Any:
            if planner is None:
                return parse_plan_from_text(task)
            plan_prompt = f"{TODO_PLANNER_INSTRUCTIONS.strip()}\n\nTask: {task.strip()}"
            result = await planner.run(plan_prompt)
            plan_obj = getattr(result, "data", None) or getattr(result, "output", None) or result
            if isinstance(plan_obj, PlanSteps):
                return plan_obj
            if isinstance(plan_obj, str):
                parsed = parse_plan_from_text(plan_obj)
                return parsed or plan_obj
            return plan_obj

        setattr(runner, "_isaac_todo_registered", True)

    async def _build_runner(
        self,
        session_id: str,
        state: SubagentSessionState,
        toolsets: list[Any] | None = None,
    ) -> None:
        try:
            executor = create_subagent_for_model(
                model_registry.current_model_id(),
                self._register_tools,
                toolsets=toolsets,
            )
            state.model_id = model_registry.current_model_id()
            state.runner = executor
            state.todo_planner = create_subagent_planner_for_model(state.model_id)
            self._attach_todo_tool(state)
            state.history = []
            state.model_error = None
            state.model_error_notified = False
        except Exception as exc:
            msg = f"Model load failed: {exc}"
            state.runner = None
            state.todo_planner = None
            state.model_error = msg
            state.model_error_notified = False
            await self.env.send_update(
                session_notification(
                    session_id,
                    update_agent_message(text_block(msg)),
                )
            )

    async def _respond_model_error(self, session_id: str, state: SubagentSessionState) -> PromptResponse:
        msg = state.model_error or "Model unavailable for this session."
        if not state.model_error_notified:
            await self.env.send_update(
                session_notification(
                    session_id,
                    update_agent_message(text_block(msg)),
                )
            )
            state.model_error_notified = True
        return PromptResponse(stop_reason="refusal")

    def _plan_from_todo_result(self, result_part: Any) -> Any | None:
        content = getattr(result_part, "content", None)
        plan_obj = content
        if isinstance(plan_obj, PlanSteps) and plan_obj.entries:
            plan_update = plan_update_from_steps(plan_obj.entries)
            if plan_update is not None:
                return plan_update

        if isinstance(plan_obj, str):
            return parse_plan_from_text(plan_obj or "")
        if isinstance(plan_obj, dict):
            return parse_plan_from_text(str(plan_obj))
        return None
