"""Default two-stage handoff prompt strategy."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Dict

from acp.helpers import session_notification, text_block, update_agent_message
from acp.schema import PromptResponse

from isaac.agent import models as model_registry
from isaac.agent.brain.strategy_runner import StrategyEnv
from isaac.agent.brain.handoff_runner import HandoffRunner
from isaac.agent.brain.strategy_base import ModelBuildError, PromptStrategy
from isaac.agent.brain.strategy_utils import create_agents_for_model
from isaac.agent.runner import register_tools as default_register_tools


@dataclass
class PromptSessionState:
    """State for a single session owned by the handoff strategy."""

    runner: Any | None = None
    planner: Any | None = None
    model_id: str | None = None
    planner_history: list[Any] = field(default_factory=list)
    executor_history: list[Any] = field(default_factory=list)
    model_error: str | None = None
    model_error_notified: bool = False


class HandoffPromptStrategy(PromptStrategy):
    """Default planning/execution hand-off strategy."""

    def __init__(
        self,
        env: StrategyEnv,
        *,
        register_tools: Callable[[Any], None] | None = None,
    ) -> None:
        self.env = env
        self._register_tools = register_tools or default_register_tools
        self._handoff_runner = HandoffRunner(env)
        self._sessions: Dict[str, PromptSessionState] = {}

    async def init_session(self, session_id: str, toolsets: list[Any], system_prompt: str | None = None) -> None:
        """Initialize per-session runner/planner state."""
        state = self._sessions.setdefault(
            session_id,
            PromptSessionState(model_id=model_registry.current_model_id()),
        )
        await self._build_runners(session_id, state, toolsets, system_prompt=system_prompt)

    async def set_session_model(
        self, session_id: str, model_id: str, toolsets: list[Any], system_prompt: str | None = None
    ) -> None:
        """Switch the backing model for a session."""
        state = self._sessions.setdefault(
            session_id,
            PromptSessionState(model_id=model_registry.current_model_id()),
        )
        try:
            executor, planner = create_agents_for_model(
                model_id, self._register_tools, toolsets=toolsets, system_prompt=system_prompt
            )
            state.runner = executor
            state.planner = planner
            state.model_id = model_id
            state.model_error = None
            state.model_error_notified = False
            state.planner_history = []
            state.executor_history = []
        except Exception as exc:
            msg = f"Model load failed: {exc}"
            state.runner = None
            state.planner = None
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
        """Run the planning/execution handoff for a prompt."""
        state = self._sessions.setdefault(
            session_id,
            PromptSessionState(model_id=model_registry.current_model_id()),
        )
        if state.runner is None or state.planner is None:
            await self._build_runners(session_id, state, toolsets=None)

        runner = state.runner
        planner = state.planner
        if state.model_error or runner is None or planner is None:
            return await self._respond_model_error(session_id, state)

        planner_history = list(state.planner_history)
        executor_history = list(state.executor_history)

        def _store_planner_messages(msgs: Any) -> None:
            try:
                state.planner_history.extend(list(msgs or []))
            except Exception:
                return

        def _store_executor_messages(msgs: Any) -> None:
            try:
                if isinstance(msgs, dict):
                    items = [msgs]
                else:
                    items = list(msgs or [])
                state.executor_history.extend(items)
            except Exception:
                return

        return await self._handoff_runner.run(
            session_id,
            prompt_text,
            planner_history=planner_history,
            executor_history=executor_history,
            cancel_event=cancel_event,
            runner=runner,
            planner=planner,
            store_planner_messages=_store_planner_messages,
            store_executor_messages=_store_executor_messages,
            record_executor_history=_store_executor_messages,
        )

    def model_id(self, session_id: str) -> str | None:
        return self._sessions.get(session_id, PromptSessionState()).model_id

    def set_session_runner(self, session_id: str, runner: Any | None) -> None:
        state = self._sessions.setdefault(
            session_id,
            PromptSessionState(model_id=model_registry.current_model_id()),
        )
        state.runner = runner
        state.model_id = state.model_id or model_registry.current_model_id()
        state.model_error = None
        state.model_error_notified = False

    def get_session_runner(self, session_id: str) -> Any | None:
        return self._sessions.get(session_id, PromptSessionState()).runner

    def session_ids(self) -> list[str]:
        return list(self._sessions.keys())

    async def _build_runners(
        self,
        session_id: str,
        state: PromptSessionState,
        toolsets: list[Any] | None = None,
        system_prompt: str | None = None,
    ) -> None:
        try:
            executor, planner = create_agents_for_model(
                model_registry.current_model_id(),
                self._register_tools,
                toolsets=toolsets,
                system_prompt=system_prompt,
            )
            state.model_id = model_registry.current_model_id()
            state.runner = executor
            state.planner = planner
            state.model_error = None
            state.model_error_notified = False
            state.planner_history = []
            state.executor_history = []
        except Exception as exc:
            msg = f"Model load failed: {exc}"
            state.runner = None
            state.planner = None
            state.model_error = msg
            state.model_error_notified = False
            await self.env.send_update(
                session_notification(
                    session_id,
                    update_agent_message(text_block(msg)),
                )
            )

    async def _respond_model_error(self, session_id: str, state: PromptSessionState) -> PromptResponse:
        """Emit a model error message (once) and end the prompt turn."""
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
