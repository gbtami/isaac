"""Single-agent prompt handler with delegate planning support."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict

from acp.helpers import session_notification, text_block, update_agent_message
from acp.schema import PromptResponse
from pydantic_ai.messages import FunctionToolResultEvent  # type: ignore

from isaac.agent import models as model_registry
from isaac.agent.brain.prompt_runner import PromptEnv, PromptRunner
from isaac.agent.brain.plan_parser import parse_plan_from_text
from isaac.agent.brain.model_errors import ModelBuildError
from isaac.agent.brain.plan_schema import PlanSteps
from isaac.agent.brain.agent_factory import create_subagent_for_model
from isaac.agent.brain.plan_updates import plan_update_from_steps, plan_with_status
from isaac.agent.runner import stream_with_runner
from isaac.agent.subagents.delegate_tools import (
    DelegateToolContext,
    reset_delegate_tool_context,
    set_delegate_tool_context,
)
from isaac.agent.tools import register_tools as default_register_tools
from isaac.agent.usage import normalize_usage


@dataclass
class SessionState:
    """State for sessions using the prompt handler."""

    runner: Any | None = None
    model_id: str | None = None
    history: list[Any] = field(default_factory=list)
    model_error: str | None = None
    model_error_notified: bool = False


class PromptHandler:
    """Single-runner prompt handler that can use delegate planning tools."""

    _MAX_HISTORY_MESSAGES = 30
    _PRESERVE_RECENT_MESSAGES = 8

    def __init__(
        self,
        env: PromptEnv,
        *,
        register_tools: Any | None = None,
    ) -> None:
        self.env = env
        self._register_tools = register_tools or default_register_tools
        self._prompt_runner = PromptRunner(env)
        self._sessions: Dict[str, SessionState] = {}

    async def init_session(self, session_id: str, toolsets: list[Any], system_prompt: str | None = None) -> None:
        state = self._sessions.setdefault(
            session_id,
            SessionState(model_id=model_registry.current_model_id()),
        )
        await self._build_runner(session_id, state, toolsets, system_prompt=system_prompt)

    async def set_session_model(
        self, session_id: str, model_id: str, toolsets: list[Any], system_prompt: str | None = None
    ) -> None:
        state = self._sessions.setdefault(
            session_id,
            SessionState(model_id=model_registry.current_model_id()),
        )
        try:
            executor = create_subagent_for_model(
                model_id, self._register_tools, toolsets=toolsets, system_prompt=system_prompt
            )
            state.runner = executor
            state.model_id = model_id
            state.history = []
            state.model_error = None
            state.model_error_notified = False
        except Exception as exc:
            msg = f"Model load failed: {exc}"
            state.runner = None
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
            SessionState(model_id=model_registry.current_model_id()),
        )
        if state.runner is None:
            await self._build_runner(session_id, state, toolsets=None)
        if state.model_error or state.runner is None:
            return await self._respond_model_error(session_id, state)

        history = self._trim_history(state.history, self._MAX_HISTORY_MESSAGES)
        plan_progress: dict[str, Any] | None = {"plan": None, "idx": 0}
        _push_chunk = self._prompt_runner._make_chunk_sender(session_id)  # type: ignore[attr-defined]
        _push_thought = self._prompt_runner._make_thought_sender(session_id)  # type: ignore[attr-defined]

        tool_trackers: Dict[str, Any] = {}
        run_command_ctx_tokens: Dict[str, Any] = {}

        def _record_history(msg: dict[str, Any]) -> None:
            content = str(msg.get("content") or "").strip()
            if not content:
                return
            role = str(msg.get("role") or "assistant")
            state.history.append({"role": role, "content": content})

        async def _maybe_capture_plan(event: Any) -> None:
            if not isinstance(event, FunctionToolResultEvent):
                return
            result_part = event.result
            tool_name = getattr(result_part, "tool_name", None) or ""
            if tool_name != "planner":
                return
            plan_update = self._plan_from_planner_result(result_part)
            if plan_update is None:
                return
            if plan_progress is not None:
                plan_progress["plan"] = plan_update
                plan_progress["idx"] = 0
            initial = plan_with_status(plan_update, active_index=0)
            await self.env.send_update(session_notification(session_id, initial))

        base_handler = self._prompt_runner._build_runner_event_handler(  # type: ignore[attr-defined]
            session_id,
            tool_trackers,
            run_command_ctx_tokens,
            plan_progress,
            record_history=_record_history,
        )

        async def _on_event(event: Any) -> bool:
            is_planner_result = (
                isinstance(event, FunctionToolResultEvent)
                and getattr(getattr(event, "result", None), "tool_name", "") == "planner"
            )
            if is_planner_result and plan_progress is not None:
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

        await self._maybe_compact_history(state)

        def _store_messages(msgs: Any) -> None:
            try:
                state.history.extend(list(msgs or []))
            except Exception:
                return

        # Provide parent permission routing to delegate tool runs in this prompt turn.
        delegate_token = set_delegate_tool_context(
            DelegateToolContext(
                session_id=session_id,
                request_run_permission=self.env.request_run_permission,
            )
        )
        try:
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
        finally:
            reset_delegate_tool_context(delegate_token)
        if response_text is None:
            return self._prompt_runner._prompt_cancel()  # type: ignore[attr-defined]
        if response_text.startswith("Provider error:"):
            msg = response_text.removeprefix("Provider error:").strip()
            await self.env.send_update(
                session_notification(
                    session_id,
                    update_agent_message(text_block(f"Model/provider error: {msg}")),
                )
            )
            return self._prompt_runner._prompt_end()  # type: ignore[attr-defined]
        if plan_progress and plan_progress.get("plan"):
            completed = plan_with_status(plan_progress["plan"], status_all="completed")
            await self.env.send_update(session_notification(session_id, completed))
        self.env.set_usage(session_id, normalize_usage(usage))
        return self._prompt_runner._prompt_end()  # type: ignore[attr-defined]

    def model_id(self, session_id: str) -> str | None:
        return self._sessions.get(session_id, SessionState()).model_id

    def set_session_runner(self, session_id: str, runner: Any | None) -> None:
        state = self._sessions.setdefault(
            session_id,
            SessionState(model_id=model_registry.current_model_id()),
        )
        state.runner = runner
        state.model_id = state.model_id or model_registry.current_model_id()
        state.model_error = None
        state.model_error_notified = False

    def session_ids(self) -> list[str]:
        return list(self._sessions.keys())

    def snapshot(self, session_id: str) -> dict[str, Any]:
        state = self._sessions.get(session_id)
        if state is None:
            return {}
        return {
            "history": list(state.history),
            "model_id": state.model_id,
        }

    async def restore_snapshot(self, session_id: str, snapshot: dict[str, Any]) -> None:
        state = self._sessions.setdefault(
            session_id,
            SessionState(model_id=model_registry.current_model_id()),
        )
        state.history = list(snapshot.get("history") or [])
        state.model_id = snapshot.get("model_id") or state.model_id

    async def _maybe_compact_history(self, state: SessionState) -> None:
        """Compact older history into a summary when it grows too large."""

        runner = state.runner
        if runner is None:
            return
        if len(state.history) <= self._MAX_HISTORY_MESSAGES:
            return

        to_compact = state.history[: -self._PRESERVE_RECENT_MESSAGES] or []
        preserved = state.history[-self._PRESERVE_RECENT_MESSAGES :] if state.history else []
        if not to_compact:
            return

        summary_prompt = (
            "Summarize the earlier conversation for future turns. "
            "List key tasks, decisions, files touched, commands run, and outstanding follow-ups."
        )

        async def _noop(_: str) -> None:
            return None

        summary_text = None
        try:
            summary_text, _ = await stream_with_runner(
                runner,
                summary_prompt,
                _noop,
                _noop,
                asyncio.Event(),
                history=to_compact,
                log_context="history_compact",
            )
        except Exception:
            summary_text = None

        summary_content = (summary_text or "").strip()
        summary_msg = {"role": "system", "content": f"Conversation summary:\n{summary_content}".strip()}
        state.history = [summary_msg] + preserved

    @staticmethod
    def _trim_history(history: list[Any], limit: int) -> list[Any]:
        if limit <= 0:
            return []
        if len(history) <= limit:
            return list(history)
        return list(history[-limit:])

    async def _build_runner(
        self,
        session_id: str,
        state: SessionState,
        toolsets: list[Any] | None = None,
        system_prompt: str | None = None,
    ) -> None:
        try:
            executor = create_subagent_for_model(
                model_registry.current_model_id(),
                self._register_tools,
                toolsets=toolsets,
                system_prompt=system_prompt,
            )
            state.model_id = model_registry.current_model_id()
            state.runner = executor
            state.history = []
            state.model_error = None
            state.model_error_notified = False
        except Exception as exc:
            msg = f"Model load failed: {exc}"
            state.runner = None
            state.model_error = msg
            state.model_error_notified = False
            await self.env.send_update(
                session_notification(
                    session_id,
                    update_agent_message(text_block(msg)),
                )
            )

    async def _respond_model_error(self, session_id: str, state: SessionState) -> PromptResponse:
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

    def _plan_from_planner_result(self, result_part: Any) -> Any | None:
        content = getattr(result_part, "content", None)
        plan_obj = content
        if isinstance(plan_obj, dict):
            if plan_obj.get("error"):
                return None
            if "content" in plan_obj:
                plan_obj = plan_obj.get("content")
        if isinstance(plan_obj, PlanSteps) and plan_obj.entries:
            plan_update = plan_update_from_steps(plan_obj.entries)
            if plan_update is not None:
                return plan_update

        if isinstance(plan_obj, str):
            return parse_plan_from_text(plan_obj or "")
        if isinstance(plan_obj, dict):
            return parse_plan_from_text(str(plan_obj))
        return None
