"""Session lifecycle helpers for prompt handling."""

from __future__ import annotations

from typing import Any

from isaac.agent.ai_types import AgentRunner, ToolRegister
from isaac.agent.brain.prompt_runner import PromptEnv

from isaac.agent import models as model_registry
from isaac.agent.brain.agent_factory import create_subagent_for_model
from isaac.agent.brain.model_errors import ModelBuildError
from isaac.agent.brain.prompt_result import PromptResult
from isaac.agent.brain.session_state import SessionState


async def set_session_model(
    *,
    env: PromptEnv,
    session_id: str,
    state: SessionState,
    model_id: str,
    register_tools: ToolRegister,
    toolsets: list[Any],
    system_prompt: str | None = None,
) -> None:
    """Swap the session model and reset its state."""

    try:
        executor: AgentRunner = create_subagent_for_model(
            model_id, register_tools, toolsets=toolsets, system_prompt=system_prompt
        )
        state.runner = executor
        state.model_id = model_id
        state.system_prompt = system_prompt
        state.history = []
        state.recent_files = []
        state.last_usage_total_tokens = None
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
) -> None:
    """Build the default runner for a session."""

    try:
        executor: AgentRunner = create_subagent_for_model(
            model_registry.current_model_id(),
            register_tools,
            toolsets=toolsets,
            system_prompt=system_prompt,
        )
        state.model_id = model_registry.current_model_id()
        state.runner = executor
        state.system_prompt = system_prompt
        state.history = []
        state.recent_files = []
        state.last_usage_total_tokens = None
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
