"""Create pydantic-ai agents for prompt handling."""

from __future__ import annotations

from typing import Any

from dotenv import load_dotenv
from pydantic_ai import Agent as PydanticAgent  # type: ignore

from isaac.agent.ai_types import AgentRunner, ModelLike, ModelSettingsLike, ToolRegister
from isaac.agent.brain.prompt import SUBAGENT_INSTRUCTIONS, SYSTEM_PROMPT
from isaac.agent.models import ENV_FILE, load_models_config, _build_provider_model


def create_subagent_for_model(
    model_id: str,
    register_tools: ToolRegister,
    toolsets: list[Any] | None = None,
    system_prompt: str | None = None,
) -> AgentRunner:
    """Build a single-runner agent for subagent mode."""

    load_dotenv(ENV_FILE, override=False)
    load_dotenv()
    config = load_models_config()
    models_cfg = config.get("models", {})
    if model_id not in models_cfg:
        raise ValueError(f"Unknown model id: {model_id}")
    model_entry = models_cfg.get(model_id, {})

    model_obj: ModelLike
    model_settings: ModelSettingsLike
    model_obj, model_settings = _build_provider_model(model_id, model_entry)

    runner: AgentRunner = PydanticAgent(
        model_obj,
        toolsets=toolsets or (),
        system_prompt=system_prompt or SYSTEM_PROMPT,
        instructions=SUBAGENT_INSTRUCTIONS,
        model_settings=model_settings,
    )
    register_tools(runner)
    return runner
