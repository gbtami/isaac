"""Session lifecycle helpers for prompt handling."""

from __future__ import annotations

import logging
from typing import Any, Callable

from isaac.agent.ai_types import AgentRunner, ToolRegister
from isaac.agent.brain.prompt_runner import PromptEnv

from isaac.agent import models as model_registry
from isaac.agent.brain.agent_factory import create_subagent_for_model
from isaac.agent.brain.compaction import (
    auto_compact_token_limit,
    estimate_history_tokens,
    maybe_compact_history,
)
from isaac.agent.brain.model_errors import ModelBuildError
from isaac.agent.brain.prompt_result import PromptResult
from isaac.agent.brain.session_state import SessionState
from isaac.log_utils import log_context as log_ctx, log_event

RunnerFactory = Callable[..., AgentRunner]
logger = logging.getLogger(__name__)


async def set_session_model(
    *,
    env: PromptEnv,
    session_id: str,
    state: SessionState,
    model_id: str,
    register_tools: ToolRegister,
    toolsets: list[Any],
    system_prompt: str | None = None,
    runner_factory: RunnerFactory | None = None,
    auto_compact_ratio: float = 0.9,
    compact_user_message_max_tokens: int = 20_000,
    max_history_messages: int = 30,
) -> None:
    """Swap the session model while preserving and right-sizing existing history.

    We intentionally keep conversation history across model switches so ACP clients
    get consistent multi-turn behavior. When switching to a smaller context-window
    model, we compact with the previous runner first so the old model can summarize
    details that might not fit in the new context budget.
    """

    try:
        previous_model_id = state.model_id
        await _maybe_compact_before_model_switch(
            env=env,
            session_id=session_id,
            state=state,
            previous_model_id=previous_model_id,
            next_model_id=model_id,
            auto_compact_ratio=auto_compact_ratio,
            compact_user_message_max_tokens=compact_user_message_max_tokens,
            max_history_messages=max_history_messages,
        )
        factory = runner_factory or create_subagent_for_model
        executor = factory(
            model_id,
            register_tools,
            toolsets=toolsets,
            system_prompt=system_prompt,
            session_mode_getter=lambda: env.session_modes.get(session_id, "ask"),
        )
        state.runner = executor
        state.model_id = model_id
        state.system_prompt = system_prompt
        state.model_error = None
        state.model_error_notified = False
    except Exception as exc:
        msg = f"Model load failed: {exc}"
        state.runner = None
        state.model_error = msg
        state.model_error_notified = False
        await env.send_notification(session_id, msg)
        raise ModelBuildError(msg) from exc


async def build_runner(
    *,
    env: PromptEnv,
    session_id: str,
    state: SessionState,
    register_tools: ToolRegister,
    toolsets: list[Any] | None = None,
    system_prompt: str | None = None,
    runner_factory: RunnerFactory | None = None,
) -> None:
    """Build the default runner for a session."""

    try:
        factory = runner_factory or create_subagent_for_model
        executor = factory(
            model_registry.current_model_id(),
            register_tools,
            toolsets=toolsets,
            system_prompt=system_prompt,
            session_mode_getter=lambda: env.session_modes.get(session_id, "ask"),
        )
        state.model_id = model_registry.current_model_id()
        state.runner = executor
        state.system_prompt = system_prompt
        state.history = []
        state.recent_files = []
        state.last_usage_total_tokens = None
        state.usage_total_tokens_since_compaction = 0
        state.last_compaction_checkpoint = None
        state.model_error = None
        state.model_error_notified = False
    except Exception as exc:
        msg = f"Model load failed: {exc}"
        state.runner = None
        state.model_error = msg
        state.model_error_notified = False
        await env.send_notification(session_id, msg)


async def respond_model_error(
    *,
    env: PromptEnv,
    session_id: str,
    state: SessionState,
) -> PromptResult:
    """Return a refusal response and notify once."""

    msg = state.model_error or "Model unavailable for this session."
    if not state.model_error_notified:
        await env.send_notification(session_id, msg)
        state.model_error_notified = True
    return PromptResult(stop_reason="refusal")


async def _maybe_compact_before_model_switch(
    *,
    env: PromptEnv,
    session_id: str,
    state: SessionState,
    previous_model_id: str | None,
    next_model_id: str,
    auto_compact_ratio: float,
    compact_user_message_max_tokens: int,
    max_history_messages: int,
) -> None:
    """Compact history before switching to a smaller model context window.

    This mirrors Codex's pre-sampling behavior: when the next model has a
    materially smaller context window, summarize the existing transcript with
    the current model first to reduce overflow risk on the next prompt.
    """

    if not previous_model_id or previous_model_id == next_model_id:
        return
    if state.runner is None or not state.history:
        return

    previous_limit = model_registry.get_context_limit(previous_model_id)
    next_limit = model_registry.get_context_limit(next_model_id)
    if not previous_limit or not next_limit:
        return
    if previous_limit <= next_limit:
        return

    next_token_limit = auto_compact_token_limit(next_limit, auto_compact_ratio)
    if next_token_limit is None:
        return

    estimated_tokens = estimate_history_tokens(state.history)
    usage_signal = max(
        estimated_tokens,
        state.last_usage_total_tokens or 0,
        state.usage_total_tokens_since_compaction,
    )
    if usage_signal <= next_token_limit:
        return

    with log_ctx(session_id=session_id, model_id=previous_model_id):
        log_event(
            logger,
            "prompt.history.compact.model_switch",
            estimated_tokens=estimated_tokens,
            usage_signal_tokens=usage_signal,
            next_model_id=next_model_id,
            previous_context_limit=previous_limit,
            next_context_limit=next_limit,
            next_token_limit=next_token_limit,
        )

    await maybe_compact_history(
        env=env,
        state=state,
        session_id=session_id,
        model_id=previous_model_id,
        max_history_messages=max_history_messages,
        auto_compact_ratio=auto_compact_ratio,
        compact_user_message_max_tokens=compact_user_message_max_tokens,
    )
