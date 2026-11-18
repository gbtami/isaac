"""Verify request routing where providers differ in their inline compaction support."""

import pytest
from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import CompactionPart, ModelMessage, ModelRequest, ModelResponse, TextPart
from pydantic_ai.models import Model, ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import RunUsage

from isaac.compaction import AutomaticCompaction


def request_for(model: Model, *history: ModelMessage) -> tuple[RunContext[None], ModelRequestContext]:
    messages = [*history, ModelRequest.user_text_prompt("Continue the task.")]
    ctx = RunContext(deps=None, model=model, usage=RunUsage(), messages=messages)
    request = ModelRequestContext(
        model=model,
        messages=messages.copy(),
        model_settings={"max_tokens": 200},
        model_request_parameters=ModelRequestParameters(),
    )
    return ctx, request


@pytest.mark.parametrize("base_url", ["https://api.openai.com/v1", "https://example.com/v1"])
async def test_inline_openai_retries_without_rejected_compaction_edit(base_url: str) -> None:
    model = OpenAIResponsesModel("gpt-5.6-luna", provider=OpenAIProvider(api_key="fake", base_url=base_url))
    ctx, request = request_for(model)
    attempts: list[object] = []

    async def handler(outgoing: ModelRequestContext) -> ModelResponse:
        settings = outgoing.model_settings or {}
        assert settings["max_tokens"] == 200
        native = settings.get("openai_context_management")
        attempts.append(native)
        if native:
            raise ModelHTTPError(status_code=400, model_name=model.model_name, body="Unknown context_management field")
        return ModelResponse(parts=[TextPart("Continue.")])

    result = await AutomaticCompaction(100_000).wrap_model_request(ctx, request_context=request, handler=handler)
    assert result.parts == [TextPart("Continue.")]
    if base_url == "https://api.openai.com/v1":
        assert attempts == [[{"type": "compaction", "compact_threshold": 85_000}], None]
    else:
        assert attempts == [None]


@pytest.mark.parametrize("signed", [False, True])
async def test_anthropic_inline_edit_is_compatible_with_history(signed: bool) -> None:
    model = AnthropicModel("claude-haiku-5-5", provider=AnthropicProvider(api_key="fake"))
    summary = CompactionPart(
        "Current project decisions.",
        provider_name="anthropic",
        provider_details={"signature": "fixture"} if signed else None,
    )
    ctx, request = request_for(model, ModelResponse(parts=[summary], provider_name="anthropic"))

    async def handler(outgoing: ModelRequestContext) -> ModelResponse:
        assert summary in outgoing.messages[0].parts
        native = (outgoing.model_settings or {}).get("anthropic_context_management")
        if signed:
            assert native is None
        else:
            assert native == {
                "edits": [{"type": "compact_20260112", "trigger": {"type": "input_tokens", "value": 85_000}}]
            }
        return ModelResponse(parts=[TextPart("Continue.")])

    await AutomaticCompaction(100_000).wrap_model_request(ctx, request_context=request, handler=handler)
