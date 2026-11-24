"""Verify request routing where providers differ in their inline compaction support."""

import pytest
from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import (
    CompactionPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models import Model, ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import RequestUsage, RunUsage
from pydantic_ai_harness.compaction import estimate_token_count

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


@pytest.mark.parametrize("summary_fails", [False, True])
async def test_compacts_a_short_history_that_nearly_fills_a_small_context(summary_fails: bool) -> None:
    summary_calls = 0

    def summarize(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal summary_calls
        summary_calls += 1
        if summary_fails:
            raise ModelHTTPError(status_code=503, model_name="local", body="Could not summarize")
        return ModelResponse(parts=[TextPart("Summary of the earlier work.")])

    model = FunctionModel(summarize)
    previous = TextPart("Earlier unrelated details about the project. " * 1250)
    active_prompt = ModelRequest.user_text_prompt("Review the cache behavior.")
    call = ModelResponse(
        parts=[ToolCallPart("shell", {"command": "pytest"}, "run-tests")],
        usage=RequestUsage(input_tokens=16_000, output_tokens=45),
    )
    tool_output = (
        "All tests passed." if summary_fails else "Detailed latest test results for the cache behavior. " * 400
    )
    result = ModelRequest(parts=[ToolReturnPart("shell", tool_output, "run-tests")])
    ctx, request = request_for(
        model,
        ModelRequest.user_text_prompt("What model are you using?"),
        ModelResponse(parts=[TextPart("I'm using Flint.")]),
        ModelRequest.user_text_prompt("Tell me about this project."),
        ModelResponse(parts=[previous]),
        active_prompt,
        call,
    )
    ctx.messages[-1] = request.messages[-1] = result

    async def handler(outgoing: ModelRequestContext) -> ModelResponse:
        assert outgoing.messages[-2:] == [call, result]
        assert any("Review the cache behavior." in str(message.parts) for message in outgoing.messages)
        assert all(previous not in message.parts for message in outgoing.messages)
        assert estimate_token_count(outgoing.messages) < 16_384 * 0.85
        has_summary = any(isinstance(part, CompactionPart) for message in outgoing.messages for part in message.parts)
        assert has_summary is not summary_fails
        return ModelResponse(parts=[TextPart("Continue.")])

    await AutomaticCompaction[None](16_384).wrap_model_request(ctx, request_context=request, handler=handler)
    assert summary_calls == 1


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

    result = await AutomaticCompaction[None](100_000).wrap_model_request(ctx, request_context=request, handler=handler)
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

    await AutomaticCompaction[None](100_000).wrap_model_request(ctx, request_context=request, handler=handler)
