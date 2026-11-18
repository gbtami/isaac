"""Prefer Core's inline compaction on supported endpoints and keep Harness as a fallback."""

import logging
from dataclasses import replace
from typing import TypeVar
from urllib.parse import urlsplit

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability, WrapModelRequestHandler
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import CompactionPart, ModelResponse
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.anthropic import AnthropicCompaction, AnthropicModel
from pydantic_ai.models.openai import OpenAICompaction, OpenAIResponsesModel
from pydantic_ai.settings import ModelSettings, merge_model_settings
from pydantic_ai_harness.compaction import (
    DEFAULT_CONTEXT_WINDOW,
    SlidingWindowCompaction,
    SummarizingCompaction,
    TieredCompaction,
    estimate_context_tokens,
    resolve_context_window,
)

logger = logging.getLogger(__name__)
_TARGET_FRACTION = 0.85
DepsT = TypeVar("DepsT")


class AutomaticCompaction(AbstractCapability[DepsT]):
    def __init__(self, context_window: int | None = None) -> None:
        self.context_window = context_window
        self.harness = TieredCompaction[DepsT](
            tiers=[SummarizingCompaction(), SlidingWindowCompaction(keep_messages=20)],
            target_fraction=_TARGET_FRACTION,
            context_window=context_window,
        )

    async def wrap_model_request(
        self,
        ctx: RunContext[DepsT],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        model = request_context.model
        window = self.context_window or resolve_context_window(model) or DEFAULT_CONTEXT_WINDOW
        native = _native_settings(ctx, request_context, int(window * _TARGET_FRACTION))
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
        return await handler(await self.harness.before_model_request(ctx, request_context))


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
