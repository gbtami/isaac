"""Single-agent prompt handler with delegate planning support."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict
from pydantic_ai import FunctionToolResultEvent  # type: ignore
from pydantic_ai.usage import UsageLimits  # type: ignore

from isaac.agent import models as model_registry
from isaac.agent.ai_types import SessionToolDeps
from isaac.agent.brain.prompt import SUBAGENT_INSTRUCTIONS, SYSTEM_PROMPT
from isaac.agent.history_types import ChatMessage
from isaac.agent.brain.plan_helpers import plan_from_planner_result
from isaac.agent.brain.plan_progress import PlanProgress
from isaac.agent.brain.memory import CodingMemory, CodingMemoryEvent
from isaac.agent.brain.prompt_runner import PromptEnv, PromptRunner
from isaac.agent.brain.prompt_result import PromptResult
from isaac.agent.brain.history_utils import extract_usage_total, select_context_history
from isaac.agent.brain.compaction import maybe_compact_history
from isaac.agent.brain.instrumentation import base_run_metadata
from isaac.agent.brain.recent_files import record_recent_file
from isaac.agent.brain.session_ops import RunnerFactory, build_runner, respond_model_error, set_session_model
from isaac.agent.brain.session_state import SessionState
from isaac.agent.runner import stream_with_runner
from isaac.agent.capabilities import (
    build_coding_memory_capability,
    build_event_stream_observer_capability,
    build_plan_progress_capabilities,
    build_recent_files_capability,
    build_task_checkpoint_capability,
)
from isaac.agent.subagents.delegate_tools import (
    DelegateToolContext,
    reset_delegate_tool_context,
    set_delegate_tool_context,
)
from isaac.agent.usage import normalize_usage
from isaac.agent.oauth.code_assist.prompt import compose_code_assist_user_prompt
from isaac.log_utils import log_context as log_ctx, log_event

logger = logging.getLogger(__name__)


class PromptHandler:
    """Single-runner prompt handler that can use delegate planning tools."""

    _MAX_HISTORY_MESSAGES = 30
    _MAX_RECENT_FILES = 5
    _RECENT_FILES_CONTEXT = 3
    _AUTO_COMPACT_RATIO = 0.9
    _COMPACT_USER_MESSAGE_MAX_TOKENS = 20_000
    _REQUEST_LIMIT = 100
    _TOOL_CALLS_LIMIT = 200

    def __init__(
        self,
        env: PromptEnv,
        *,
        runner_factory: RunnerFactory | None = None,
    ) -> None:
        self.env = env
        self._runner_factory = runner_factory
        self._prompt_runner = PromptRunner(env)
        self._sessions: Dict[str, SessionState] = {}

    def _prepare_prompt_text(self, state: SessionState, prompt_text: str) -> str:
        model_id = state.model_id or ""
        model_entry = model_registry.list_models().get(model_id, {})
        provider = str(model_entry.get("provider") or "").lower()
        if provider == "code-assist" and not state.history:
            system_prompt = state.system_prompt or SYSTEM_PROMPT
            return compose_code_assist_user_prompt(system_prompt, prompt_text, SUBAGENT_INSTRUCTIONS)
        return prompt_text

    async def init_session(self, session_id: str, toolsets: list[Any], system_prompt: str | None = None) -> None:
        state = self._sessions.setdefault(
            session_id,
            SessionState(model_id=model_registry.current_model_id()),
        )
        await build_runner(
            env=self.env,
            session_id=session_id,
            state=state,
            toolsets=toolsets,
            system_prompt=system_prompt,
            runner_factory=self._runner_factory,
        )

    async def set_session_model(
        self, session_id: str, model_id: str, toolsets: list[Any], system_prompt: str | None = None
    ) -> None:
        state = self._sessions.setdefault(
            session_id,
            SessionState(model_id=model_registry.current_model_id()),
        )
        await set_session_model(
            env=self.env,
            session_id=session_id,
            state=state,
            model_id=model_id,
            toolsets=toolsets,
            system_prompt=system_prompt,
            runner_factory=self._runner_factory,
            auto_compact_ratio=self._AUTO_COMPACT_RATIO,
            compact_user_message_max_tokens=self._COMPACT_USER_MESSAGE_MAX_TOKENS,
            max_history_messages=self._MAX_HISTORY_MESSAGES,
        )

    async def handle_prompt(
        self,
        session_id: str,
        prompt_text: str,
        cancel_event: asyncio.Event,
    ) -> PromptResult:
        state = self._sessions.setdefault(
            session_id,
            SessionState(model_id=model_registry.current_model_id()),
        )
        if state.runner is None:
            await build_runner(
                env=self.env,
                session_id=session_id,
                state=state,
                toolsets=None,
                runner_factory=self._runner_factory,
            )
        if state.model_error or state.runner is None:
            return await respond_model_error(env=self.env, session_id=session_id, state=state)

        with log_ctx(session_id=session_id, model_id=state.model_id):
            log_event(
                logger,
                "prompt.handle.start",
                prompt_preview=prompt_text[:160].replace("\n", "\\n"),
            )

        await maybe_compact_history(
            env=self.env,
            state=state,
            session_id=session_id,
            model_id=state.model_id,
            max_history_messages=self._MAX_HISTORY_MESSAGES,
            auto_compact_ratio=self._AUTO_COMPACT_RATIO,
            compact_user_message_max_tokens=self._COMPACT_USER_MESSAGE_MAX_TOKENS,
        )
        history = select_context_history(
            state.history,
            current_prompt=prompt_text,
            context_limit=model_registry.get_context_limit(state.model_id),
        )
        plan_progress = PlanProgress()
        _push_thought = self._prompt_runner._make_thought_sender(session_id)  # type: ignore[attr-defined]

        # Keep tool/plan/thought updates streaming, but emit assistant text only once
        # at end-of-turn to avoid duplicate/provisional chunk rendering artifacts.
        async def _drop_chunk(_: str) -> None:
            return None

        def _record_history(msg: ChatMessage) -> None:
            content = str(msg.get("content") or "").strip()
            if not content:
                return
            role = str(msg.get("role") or "assistant")
            item: ChatMessage = {
                "role": role,
                "content": content,
                "source": str(msg.get("source") or "tool_summary"),
            }
            for key in ("tool_name", "tool_kind", "synthetic", "checkpoint"):
                if key in msg:
                    item[key] = msg[key]
            state.history.append(item)

        def _record_memory(event: CodingMemoryEvent) -> None:
            state.coding_memory.append(event)

        async def _maybe_capture_plan(event: Any) -> None:
            if not isinstance(event, FunctionToolResultEvent):
                return
            result_part = getattr(event, "result", None) or getattr(event, "part", None)
            tool_name = getattr(result_part, "tool_name", None) or ""
            if tool_name != "planner":
                return
            plan_update = plan_from_planner_result(result_part)
            if plan_update is None:
                return
            plan_update = plan_progress.set_plan(plan_update)
            await self.env.send_plan_update(session_id, plan_update, None, None, list(plan_progress.statuses))

        base_handler = self._prompt_runner._build_runner_event_handler(  # type: ignore[attr-defined]
            session_id,
            plan_progress=None,
            record_history=_record_history,
            record_memory=_record_memory,
        )

        async def _on_event(event: Any) -> bool:
            handled = await base_handler(event)
            await _maybe_capture_plan(event)
            record_recent_file(state.recent_files, event, self._MAX_RECENT_FILES)
            return handled

        run_capabilities = [build_event_stream_observer_capability(_on_event)]
        run_capabilities.extend(
            build_plan_progress_capabilities(
                plan_progress,
                session_id=session_id,
                send_plan_update=self.env.send_plan_update,
            )
        )
        context_limit = model_registry.get_context_limit(state.model_id)
        task_checkpoint_capability = build_task_checkpoint_capability(
            state.coding_memory,
            current_prompt=prompt_text,
            context_limit=context_limit,
        )
        if task_checkpoint_capability is not None:
            run_capabilities.append(task_checkpoint_capability)
        coding_memory_capability = build_coding_memory_capability(
            state.coding_memory,
            current_prompt=prompt_text,
            context_limit=context_limit,
        )
        if coding_memory_capability is not None:
            run_capabilities.append(coding_memory_capability)
        recent_files_capability = build_recent_files_capability(state.recent_files, self._RECENT_FILES_CONTEXT)
        if recent_files_capability is not None:
            run_capabilities.append(recent_files_capability)

        raw_prompt_text = prompt_text
        prompt_text = self._prepare_prompt_text(state, raw_prompt_text)
        # Persist the original user intent (not provider-specific bootstrap text)
        # so future turns and compaction keep clean, user-authored context only.
        state.history.append({"role": "user", "content": raw_prompt_text, "source": "user"})

        async def _request_tool_approval(tool_call_id: str, tool_name: str, args: dict[str, Any]) -> bool:
            if tool_name != "run_command":
                return True
            mode = self.env.session_modes.get(session_id, "ask")
            if mode != "ask":
                return True
            command = str(args.get("command") or "")
            cwd = args.get("cwd")
            return await self.env.request_run_permission(
                session_id=session_id,
                tool_call_id=tool_call_id,
                command=command,
                cwd=cwd if isinstance(cwd, str) or cwd is None else str(cwd),
            )

        # Provide parent permission routing to delegate tool runs in this prompt turn.
        session_cwd = self.env.session_cwd(session_id)
        tool_deps = None
        if session_cwd is not None:
            tool_deps = SessionToolDeps(
                session_id=session_id,
                cwd=session_cwd,
                additional_directories=self.env.session_additional_directories(session_id),
                mode=self.env.session_modes.get(session_id, "ask"),
                model_id=state.model_id or "",
            )
        delegate_token = set_delegate_tool_context(
            DelegateToolContext(
                session_id=session_id,
                request_run_permission=self.env.request_run_permission,
                send_update=self.env.send_protocol_update,
                mode_getter=lambda: self.env.session_modes.get(session_id, "ask"),
                cwd=session_cwd,
                additional_directories=self.env.session_additional_directories(session_id),
                model_id=state.model_id or "",
            )
        )
        try:
            response_text, usage = await stream_with_runner(
                state.runner,
                prompt_text,
                _drop_chunk,
                _push_thought,
                cancel_event,
                history=history,
                capabilities=run_capabilities,
                deps=tool_deps,
                log_context="subagent",
                request_tool_approval=_request_tool_approval,
                usage_limits=UsageLimits(
                    request_limit=self._REQUEST_LIMIT,
                    tool_calls_limit=self._TOOL_CALLS_LIMIT,
                ),
                metadata=base_run_metadata(
                    component="isaac.agent.turn",
                    model_id=state.model_id or "unknown",
                    extra={
                        "session_id": session_id,
                        "mode": self.env.session_modes.get(session_id, "ask"),
                    },
                ),
            )
        finally:
            reset_delegate_tool_context(delegate_token)
        if response_text is None:
            with log_ctx(session_id=session_id, model_id=state.model_id):
                log_event(logger, "prompt.handle.cancelled")
            return self._prompt_runner._prompt_cancel()  # type: ignore[attr-defined]
        if response_text.startswith("Provider timeout:"):
            msg = response_text.removeprefix("Provider timeout:").strip()
            with log_ctx(session_id=session_id, model_id=state.model_id):
                log_event(logger, "prompt.handle.timeout", level=logging.WARNING, error=msg)
            await self.env.send_notification(session_id, f"Model/provider timeout: {msg}")
            return self._prompt_runner._prompt_end()  # type: ignore[attr-defined]
        if response_text.startswith("Provider error:"):
            msg = response_text.removeprefix("Provider error:").strip()
            with log_ctx(session_id=session_id, model_id=state.model_id):
                log_event(logger, "prompt.handle.error", level=logging.WARNING, error=msg)
            await self.env.send_notification(session_id, f"Model/provider error: {msg}")
            return self._prompt_runner._prompt_end()  # type: ignore[attr-defined]
        if response_text:
            payload = response_text if response_text.endswith("\n") else f"{response_text}\n"
            await self.env.send_message_chunk(session_id, payload)
            state.history.append({"role": "assistant", "content": response_text, "source": "assistant"})
        # Plan steps are no longer auto-completed at end of turn. The model
        # reports concrete progress through the run-scoped mark_plan_step tool,
        # avoiding false completion from exploratory or failed work.
        with log_ctx(session_id=session_id, model_id=state.model_id):
            log_event(
                logger,
                "prompt.history.updated",
                level=logging.DEBUG,
                history_len=len(state.history),
                last_preview=(state.history[-1].get("content", "") if state.history else "")[:120].replace("\n", "\\n"),
            )
        normalized_usage = normalize_usage(usage)
        state.last_usage_total_tokens = extract_usage_total(normalized_usage)
        if state.last_usage_total_tokens is not None:
            # Keep a compaction-era cumulative usage signal. We reset it when
            # compaction runs, so this tracks growth pressure between compactions.
            state.usage_total_tokens_since_compaction += max(state.last_usage_total_tokens, 0)
        self.env.set_usage(session_id, normalized_usage)
        with log_ctx(session_id=session_id, model_id=state.model_id):
            log_event(logger, "prompt.handle.complete")
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
        state.usage_total_tokens_since_compaction = 0
        state.last_compaction_checkpoint = None
        state.coding_memory = CodingMemory()

    def close_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def session_ids(self) -> list[str]:
        return list(self._sessions.keys())

    def snapshot(self, session_id: str) -> dict[str, Any]:
        state = self._sessions.get(session_id)
        if state is None:
            return {}
        return {
            "history": list(state.history),
            "model_id": state.model_id,
            "recent_files": list(state.recent_files),
            "coding_memory": state.coding_memory.model_dump_jsonable(),
            "usage_total_tokens_since_compaction": state.usage_total_tokens_since_compaction,
            "last_compaction_checkpoint": state.last_compaction_checkpoint,
        }

    async def restore_snapshot(self, session_id: str, snapshot: dict[str, Any]) -> None:
        state = self._sessions.setdefault(
            session_id,
            SessionState(model_id=model_registry.current_model_id()),
        )
        state.history = list(snapshot.get("history") or [])
        state.model_id = snapshot.get("model_id") or state.model_id
        state.recent_files = list(snapshot.get("recent_files") or [])
        state.coding_memory = CodingMemory.from_snapshot(snapshot.get("coding_memory"))
        usage_since_compaction = snapshot.get("usage_total_tokens_since_compaction")
        state.usage_total_tokens_since_compaction = (
            int(usage_since_compaction) if isinstance(usage_since_compaction, int) else 0
        )
        checkpoint = snapshot.get("last_compaction_checkpoint")
        state.last_compaction_checkpoint = checkpoint if isinstance(checkpoint, dict) else None
