from __future__ import annotations

import pytest
from pydantic_ai import Agent as PydanticAgent, messages as ai_messages  # type: ignore
from pydantic_ai.models.test import TestModel  # type: ignore

from isaac.agent.brain.history_processors import sanitize_message_history


@pytest.mark.asyncio
async def test_sanitize_message_history_preserves_request_instructions_and_metadata() -> None:
    request = ai_messages.ModelRequest(
        parts=[
            ai_messages.SystemPromptPart(content="  system prompt  "),
            ai_messages.UserPromptPart(content="  user prompt  "),
            ai_messages.UserPromptPart(content="   "),
        ],
        instructions="codex instructions",
        run_id="run-123",
        metadata={"trace_id": "abc"},
    )

    cleaned = await sanitize_message_history([request])

    assert len(cleaned) == 1
    cleaned_request = cleaned[0]
    assert isinstance(cleaned_request, ai_messages.ModelRequest)
    assert cleaned_request.instructions == "codex instructions"
    assert cleaned_request.run_id == "run-123"
    assert cleaned_request.metadata == {"trace_id": "abc"}
    assert len(cleaned_request.parts) == 2
    assert isinstance(cleaned_request.parts[0], ai_messages.SystemPromptPart)
    assert cleaned_request.parts[0].content == "system prompt"
    assert isinstance(cleaned_request.parts[1], ai_messages.UserPromptPart)
    assert cleaned_request.parts[1].content == "user prompt"


@pytest.mark.asyncio
async def test_sanitize_message_history_preserves_response_metadata() -> None:
    response = ai_messages.ModelResponse(
        parts=[
            ai_messages.TextPart(content="  answer  "),
            ai_messages.TextPart(content="   "),
        ],
        model_name="gpt-5.3-codex",
        provider_name="openai-codex",
        provider_response_id="resp_123",
        run_id="run-xyz",
        metadata={"step": 1},
    )

    cleaned = await sanitize_message_history([response])

    assert len(cleaned) == 1
    cleaned_response = cleaned[0]
    assert isinstance(cleaned_response, ai_messages.ModelResponse)
    assert cleaned_response.model_name == "gpt-5.3-codex"
    assert cleaned_response.provider_name == "openai-codex"
    assert cleaned_response.provider_response_id == "resp_123"
    assert cleaned_response.run_id == "run-xyz"
    assert cleaned_response.metadata == {"step": 1}
    assert len(cleaned_response.parts) == 1
    assert isinstance(cleaned_response.parts[0], ai_messages.TextPart)
    assert cleaned_response.parts[0].content == "answer"


def test_history_capabilities_are_exposed_as_pydantic_ai_capabilities() -> None:
    from isaac.agent.capabilities import (
        build_base_capabilities,
        build_history_sanitizer_capability,
        build_system_prompt_capability,
    )

    assert type(build_system_prompt_capability()).__name__ == "ReinjectSystemPrompt"
    assert type(build_history_sanitizer_capability()).__name__ == "ProcessHistory"

    base_capabilities = build_base_capabilities(lambda: "ask")
    names = [type(capability).__name__ for capability in base_capabilities]
    assert names[:2] == ["ReinjectSystemPrompt", "ProcessHistory"]


@pytest.mark.asyncio
async def test_base_capabilities_reinject_current_system_prompt() -> None:
    from isaac.agent.capabilities import build_base_capabilities

    class CapturingTestModel(TestModel):
        captured_messages: list[ai_messages.ModelMessage]

        def _request(self, messages, model_settings, model_request_parameters):  # type: ignore[no-untyped-def]
            self.captured_messages = list(messages)
            return super()._request(messages, model_settings, model_request_parameters)

    model = CapturingTestModel(custom_output_text="ok")
    agent = PydanticAgent(
        model,
        system_prompt="CURRENT_SYSTEM_PROMPT",
        capabilities=build_base_capabilities(lambda: "ask"),
    )

    await agent.run(
        "new prompt",
        message_history=[
            ai_messages.ModelRequest(
                parts=[
                    ai_messages.SystemPromptPart(content="OLD_SYSTEM_PROMPT"),
                    ai_messages.UserPromptPart(content="previous user"),
                ]
            )
        ],
    )

    captured = model.captured_messages
    system_parts = [
        part
        for message in captured
        if isinstance(message, ai_messages.ModelRequest)
        for part in message.parts
        if isinstance(part, ai_messages.SystemPromptPart)
    ]
    assert [part.content for part in system_parts] == ["CURRENT_SYSTEM_PROMPT"]


def test_optional_harness_capabilities_are_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    from isaac.agent.capabilities import build_optional_harness_capabilities

    monkeypatch.delenv("ISAAC_HARNESS_FILESYSTEM", raising=False)
    monkeypatch.delenv("ISAAC_HARNESS_SHELL", raising=False)
    monkeypatch.delenv("ISAAC_HARNESS_CODE_MODE", raising=False)

    assert build_optional_harness_capabilities() == []
