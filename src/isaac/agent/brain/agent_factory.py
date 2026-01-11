"""Create pydantic-ai agents for prompt handling."""

from __future__ import annotations

from typing import Any

from dotenv import load_dotenv
from pydantic_ai import Agent as PydanticAgent  # type: ignore

from isaac.agent.ai_types import AgentRunner, ModelLike, ModelSettingsLike, ToolRegister
from isaac.agent.brain.prompt import SUBAGENT_INSTRUCTIONS, SYSTEM_PROMPT
from isaac.agent.oauth.code_assist.prompt import antigravity_instructions
from isaac.agent.oauth.openai_codex.prompt import codex_instructions
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

    provider = str(model_entry.get("provider") or "").lower()
    effective_system_prompt = SYSTEM_PROMPT if system_prompt is None else system_prompt
    instructions = SUBAGENT_INSTRUCTIONS
    if provider == "openai-codex":
        # Codex backend rejects custom instructions; use Codex CLI prompt instead.
        instructions = codex_instructions()
        effective_system_prompt = ""
    elif provider == "code-assist":
        instructions = antigravity_instructions()
        effective_system_prompt = ""

    runner: AgentRunner = PydanticAgent(
        model_obj,
        toolsets=toolsets or (),
        system_prompt=effective_system_prompt,
        instructions=instructions,
        model_settings=model_settings,
    )
    register_tools(runner)
    return runner
