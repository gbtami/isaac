"""Prefer Core's inline compaction on supported endpoints and keep Harness as a fallback."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Never, TypeVar
from urllib.parse import urlsplit

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability, AgentNode, WrapModelRequestHandler
from pydantic_ai.exceptions import ModelHTTPError, OutputTokenLimitExceeded, UsageLimitExceeded
from pydantic_ai.messages import CompactionPart, ModelMessage, ModelRequest, ModelResponse, UserPromptPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.anthropic import AnthropicCompaction, AnthropicModel
from pydantic_ai.models.openai import OpenAICompaction, OpenAIResponsesModel
from pydantic_ai.settings import ModelSettings, merge_model_settings
from pydantic_ai_harness.compaction import (
    DEFAULT_CONTEXT_WINDOW,
    ClearToolResults,
    SlidingWindowCompaction,
    SummarizingCompaction,
    TieredCompaction,
    estimate_context_tokens,
    resolve_context_window,
)

logger = logging.getLogger(__name__)
_TARGET_FRACTION = 0.85
DepsT = TypeVar("DepsT")


class _KeepCurrentPrompt(SlidingWindowCompaction[DepsT]):
    """Retain the current task when the sliding window drops its original request."""

    async def compact(self, messages: list[ModelMessage], ctx: RunContext[DepsT]) -> list[ModelMessage]:
        kept = await super().compact(messages, ctx)
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if isinstance(message, ModelRequest) and (
                prompts := [p for p in message.parts if isinstance(p, UserPromptPart)]
            ):
                if message in kept:
                    break
                # Keep the latest task ahead of any surviving tool calls/results.
                tail = {id(item) for item in messages[index + 1 :]}
                insert_at = next((i for i, item in enumerate(kept) if id(item) in tail), len(kept))
                return [*kept[:insert_at], replace(message, parts=list(prompts)), *kept[insert_at:]]
        return kept


class AutomaticCompaction(AbstractCapability[DepsT]):
    def __init__(self, context_window: int | None = None) -> None:
        self.context_window = context_window

    async def on_node_run_error(self, ctx: RunContext[DepsT], *, node: AgentNode[DepsT], error: Exception) -> Never:
        if isinstance(error, OutputTokenLimitExceeded):
            # Harness commits completed work and sends ACP max_tokens for this usage limit.
            raise UsageLimitExceeded("output_tokens_limit reached before the model finished its response") from error
        raise error

    async def wrap_model_request(
        self,
        ctx: RunContext[DepsT],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        model = request_context.model
        window = self.context_window or resolve_context_window(model) or DEFAULT_CONTEXT_WINDOW
        target = max(1, int(window * _TARGET_FRACTION))
        native = _native_settings(ctx, request_context, target)
        # Give inline compaction room to run first. Use Harness for sudden jumps near the hard limit.
        if (
            native
            and estimate_context_tokens(
                request_context.messages, model_request_parameters=request_context.model_request_parameters
            )
            <= window * 0.95
        ):
            try:
                return await handler(
                    replace(
                        request_context, model_settings=merge_model_settings(request_context.model_settings, native)
                    )
                )
            except ModelHTTPError as error:
                # Retry only a request rejected for its compaction settings before it streams.
                if error.status_code not in (400, 422) or "context_management" not in str(error.body).lower():
                    raise
                logger.debug("Inline compaction request rejected by %s. Using Harness.", model.model_id)
        # A handful of long tool results or reasoning messages can fill a small context.
        harness = TieredCompaction[DepsT](
            tiers=[
                ClearToolResults(keep_pairs=2),
                SummarizingCompaction(keep_tokens=max(1, window // 4), keep_messages=1, keep_user_messages=True),
                _KeepCurrentPrompt(keep_tokens=max(1, window // 8)),
                _KeepCurrentPrompt(keep_messages=1),
            ],
            target_tokens=target,
        )
        return await handler(await harness.before_model_request(ctx, request_context))


def _native_settings(ctx: RunContext[DepsT], request: ModelRequestContext, threshold: int) -> ModelSettings | None:
    model = request.model
    endpoint = urlsplit(model.base_url or "")
    if endpoint.scheme != "https":
        return None
    capability: AbstractCapability[DepsT]
    if isinstance(model, OpenAIResponsesModel) and endpoint.hostname in {"api.openai.com", "chatgpt.com"}:
        capability = OpenAICompaction(token_threshold=max(1, threshold))
    elif (
        isinstance(model, AnthropicModel)
        and endpoint.hostname == "api.anthropic.com"
        and model.profile.get("anthropic_supports_on_demand_compaction", False)
        and threshold >= 50_000
    ):
        # Anthropic rejects its inline edit alongside the signed block from a manual native /compact.
        if any(
            isinstance(message, ModelResponse)
            and isinstance(part, CompactionPart)
            and (part.provider_details or {}).get("signature") is not None
            and model.accepts_stateless_artifacts(message, part)
            for message in request.messages
            for part in message.parts
        ):
            return None
        capability = AnthropicCompaction(token_threshold=threshold)
    else:
        return None
    settings = capability.get_model_settings()
    assert callable(settings)
    return settings(replace(ctx, model_settings=request.model_settings))
