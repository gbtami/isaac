"""Run tools with validation and retry semantics."""

from __future__ import annotations

import inspect
import logging
from typing import Any

from pydantic import ValidationError

from isaac.agent.tools.registry import TOOL_ARG_MODELS, TOOL_HANDLERS, TOOL_REQUIRED_ARGS
from isaac.log_utils import log_event

logger = logging.getLogger(__name__)


async def run_tool(function_name: str, ctx: Any | None = None, **kwargs: Any) -> dict:
    """Run a tool by name with pydantic validation and pydantic-ai retries."""
    handler = TOOL_HANDLERS.get(function_name)
    if not handler:
        return {"content": None, "error": f"Unknown tool function: {function_name}"}

    args_model = TOOL_ARG_MODELS.get(function_name)
    required = TOOL_REQUIRED_ARGS.get(function_name, [])
    missing_required = [name for name in required if name not in kwargs or kwargs.get(name) in ("", None)]
    if missing_required:
        msg = f"Missing required arguments: {', '.join(missing_required)}"
        if ctx is not None:
            from pydantic_ai.exceptions import ToolRetryError  # type: ignore
            from pydantic_ai.messages import RetryPromptPart  # type: ignore

            tool_call_id = getattr(ctx, "tool_call_id", None)
            log_event(
                logger,
                "tool.retry.missing_args",
                level=logging.WARNING,
                tool=function_name,
                tool_call_id=tool_call_id,
                missing=missing_required,
                args=kwargs,
            )
            raise ToolRetryError(
                RetryPromptPart(
                    content=msg,
                    tool_name=function_name,
                    tool_call_id=tool_call_id or None,
                )
            )
        return {"content": None, "error": msg}

    if args_model is not None:
        try:
            parsed = args_model.model_validate(kwargs)  # type: ignore[attr-defined]
            call_kwargs: dict[str, Any] = parsed.model_dump()  # type: ignore[attr-defined]
        except ValidationError as exc:
            msg = f"Invalid arguments: {exc}"
            if ctx is not None:
                from pydantic_ai.exceptions import ToolRetryError  # type: ignore
                from pydantic_ai.messages import RetryPromptPart  # type: ignore

                tool_call_id = getattr(ctx, "tool_call_id", None)
                log_event(
                    logger,
                    "tool.retry.validation",
                    level=logging.WARNING,
                    tool=function_name,
                    tool_call_id=tool_call_id,
                    error=str(exc),
                    args=kwargs,
                )
                raise ToolRetryError(
                    RetryPromptPart(
                        content=msg,
                        tool_name=function_name,
                        tool_call_id=tool_call_id or None,
                    )
                )
            return {"content": None, "error": msg}
    else:
        call_kwargs = dict(kwargs)

    try:
        sig = inspect.signature(handler)
    except Exception:  # pragma: no cover - defensive
        sig = inspect.signature(lambda: None)

    if "ctx" in sig.parameters:
        call_kwargs["ctx"] = ctx
    filtered_kwargs = {k: v for k, v in call_kwargs.items() if k in sig.parameters}

    try:
        result = await handler(**filtered_kwargs)
    except Exception as exc:
        # Let pydantic-ai retry tooling errors when requested.
        try:
            from pydantic_ai.exceptions import ToolRetryError  # type: ignore

            if isinstance(exc, ToolRetryError):
                raise
        except Exception:
            pass
        # Drop unexpected args (e.g., from LLM hallucinations) and retry once.
        try:
            filtered = {k: v for k, v in {**kwargs, "ctx": ctx}.items() if k in sig.parameters}
            result = await handler(**filtered)
        except Exception:
            return {"content": None, "error": str(exc)}

    if isinstance(result, dict) and result.get("content") is None:
        result["content"] = ""
    return result
