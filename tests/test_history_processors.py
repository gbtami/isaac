from __future__ import annotations

import pytest
from pydantic_ai import messages as ai_messages  # type: ignore

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
