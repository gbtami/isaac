"""Create pydantic-ai agents for prompt handling."""

from __future__ import annotations

from typing import Any, Callable

from pydantic_ai import Agent as PydanticAgent  # type: ignore
from pydantic_ai import DeferredToolRequests  # type: ignore

from isaac.agent.ai_types import AgentRunner, ModelLike, ModelSettingsLike
from isaac.agent.brain.instrumentation import base_run_metadata
from isaac.agent.brain.prompt import SUBAGENT_INSTRUCTIONS, SYSTEM_PROMPT
from isaac.agent.capabilities import build_base_capabilities
from isaac.agent.tools import build_isaac_tools_capability
from isaac.agent.oauth.code_assist.prompt import code_assist_instructions
from isaac.agent.models import load_models_config, load_runtime_env, _build_provider_model


def create_subagent_for_model(
    model_id: str,
    toolsets: list[Any] | None = None,
    system_prompt: str | None = None,
    session_mode_getter: Callable[[], str] | None = None,
) -> AgentRunner:
    """Build a single-runner agent for subagent mode."""

    load_runtime_env()
    config = load_models_config()
    models_cfg = config.get("models", {})
    if model_id not in models_cfg:
        raise ValueError(f"Unknown model id: {model_id}")
    model_entry = models_cfg.get(model_id, {})

    model_obj: ModelLike
    model_settings: ModelSettingsLike
    model_obj, model_settings = _build_provider_model(model_id, model_entry)

    provider = str(model_entry.get("provider") or "").lower()
    effective_system_prompt = SYSTEM_PROMPT if system_prompt is None else system_prompt
    instructions = SUBAGENT_INSTRUCTIONS
    if provider == "code-assist":
        instructions = code_assist_instructions()
        effective_system_prompt = ""

    mode_getter = session_mode_getter or (lambda: "ask")
    capabilities = build_base_capabilities(mode_getter)
    capabilities.append(build_isaac_tools_capability())
    effective_toolsets = list(toolsets or ())
    runner: AgentRunner = PydanticAgent(
        model_obj,
        output_type=[str, DeferredToolRequests],
        toolsets=effective_toolsets,
        system_prompt=effective_system_prompt,
        instructions=instructions,
        model_settings=model_settings,
        capabilities=capabilities,
        metadata=base_run_metadata(component="isaac.agent.main", model_id=model_id),
    )
    return runner
