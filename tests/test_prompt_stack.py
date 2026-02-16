from __future__ import annotations

from isaac.agent.brain import agent_factory
from isaac.agent.brain.prompt import SUBAGENT_INSTRUCTIONS, SYSTEM_PROMPT
from isaac.agent.oauth.code_assist.prompt import antigravity_instructions


class _CapturingAgent:
    def __init__(self, *_: object, **kwargs: object) -> None:
        self.system_prompt = kwargs.get("system_prompt")
        self.instructions = kwargs.get("instructions")


def _build_model(*_: object, **__: object) -> tuple[object, None]:
    return object(), None


def test_isaac_prompts_do_not_include_foreign_codex_tool_names() -> None:
    prompt_text = f"{SYSTEM_PROMPT}\n{SUBAGENT_INSTRUCTIONS}".lower()
    forbidden_names = (
        "apply_patch",
        "update_plan",
        "write_stdin",
        "multi_tool_use.parallel",
    )
    for name in forbidden_names:
        assert name not in prompt_text


def test_openai_codex_uses_standard_isaac_prompt_stack(monkeypatch) -> None:
    monkeypatch.setattr(agent_factory, "PydanticAgent", _CapturingAgent)
    monkeypatch.setattr(agent_factory, "load_runtime_env", lambda: None)
    monkeypatch.setattr(
        agent_factory,
        "load_models_config",
        lambda: {
            "models": {
                "openai-codex:gpt-5.3-codex": {
                    "provider": "openai-codex",
                    "model": "gpt-5.3-codex",
                }
            }
        },
    )
    monkeypatch.setattr(agent_factory, "_build_provider_model", _build_model)

    runner = agent_factory.create_subagent_for_model(
        "openai-codex:gpt-5.3-codex",
        register_tools=lambda _runner: None,
    )

    assert runner.system_prompt == SYSTEM_PROMPT
    assert runner.instructions == SUBAGENT_INSTRUCTIONS


def test_code_assist_still_uses_provider_specific_prompt_stack(monkeypatch) -> None:
    monkeypatch.setattr(agent_factory, "PydanticAgent", _CapturingAgent)
    monkeypatch.setattr(agent_factory, "load_runtime_env", lambda: None)
    monkeypatch.setattr(
        agent_factory,
        "load_models_config",
        lambda: {
            "models": {
                "code-assist:gemini": {
                    "provider": "code-assist",
                    "model": "gemini-2.5-pro",
                }
            }
        },
    )
    monkeypatch.setattr(agent_factory, "_build_provider_model", _build_model)

    runner = agent_factory.create_subagent_for_model(
        "code-assist:gemini",
        register_tools=lambda _runner: None,
    )

    assert runner.system_prompt == ""
    assert runner.instructions == antigravity_instructions()
