"""Run tools with validation and retry semantics."""

from __future__ import annotations

import inspect
import logging
from typing import Any

from pydantic import ValidationError
from pydantic_ai.exceptions import ToolRetryError  # type: ignore
from pydantic_ai.messages import RetryPromptPart  # type: ignore

from isaac.agent.ai_types import ToolContext
from isaac.agent.tools.registry import TOOL_ARG_MODELS, TOOL_HANDLERS
from isaac.log_utils import log_event

logger = logging.getLogger(__name__)


def _tool_call_id(ctx: ToolContext | None) -> str | None:
    raw = getattr(ctx, "tool_call_id", None) if ctx is not None else None
    return str(raw) if raw else None


def _raise_tool_retry(
    *,
    function_name: str,
    ctx: ToolContext,
    message: str,
    reason: str,
    args: dict[str, Any],
) -> None:
    """Ask Pydantic AI to retry the tool call with corrected arguments."""

    tool_call_id = _tool_call_id(ctx)
    log_event(
        logger,
        f"tool.retry.{reason}",
        level=logging.WARNING,
        tool=function_name,
        tool_call_id=tool_call_id,
        args=args,
    )
    raise ToolRetryError(
        RetryPromptPart(
            content=message,
            tool_name=function_name,
            tool_call_id=tool_call_id,
        )
    )


def _accepted_kwargs(
    handler: Any,
    validated_kwargs: dict[str, Any],
    *,
    runtime_kwargs: dict[str, Any],
    ctx: ToolContext | None,
) -> dict[str, Any]:
    """Keep only arguments accepted by the concrete tool handler.

    The pydantic argument models intentionally describe the model-facing tool
    contract. Runtime-only values such as the ACP session cwd are not exposed to
    the model, but they still need to reach handlers that accept them.
    """

    try:
        parameters = inspect.signature(handler).parameters
    except (TypeError, ValueError):  # pragma: no cover - defensive
        parameters = {}

    accepted = {name: value for name, value in validated_kwargs.items() if name in parameters}
    for name, value in runtime_kwargs.items():
        if name in parameters and name not in accepted:
            accepted[name] = value
    if "ctx" in parameters:
        accepted["ctx"] = ctx
    return accepted


async def run_tool(function_name: str, ctx: ToolContext | None = None, **kwargs: Any) -> dict[str, Any]:
    """Run a registered Isaac tool.

    Pydantic AI validates normal model tool calls from the Tool wrappers built in
    ``registration.py``. This function is the shared execution path for those
    wrappers and direct ACP tool-call blocks, so it keeps one pydantic validation
    pass for direct calls and lets Pydantic AI retry invalid model calls.
    """

    handler = TOOL_HANDLERS.get(function_name)
    if not handler:
        return {"content": None, "error": f"Unknown tool function: {function_name}"}

    args_model = TOOL_ARG_MODELS.get(function_name)
    raw_kwargs = dict(kwargs)
    if args_model is not None:
        try:
            parsed = args_model.model_validate(raw_kwargs)  # type: ignore[attr-defined]
            call_kwargs: dict[str, Any] = parsed.model_dump()  # type: ignore[attr-defined]
        except ValidationError as exc:
            msg = f"Invalid arguments: {exc}"
            if ctx is not None:
                _raise_tool_retry(
                    function_name=function_name,
                    ctx=ctx,
                    message=msg,
                    reason="validation",
                    args=raw_kwargs,
                )
            return {"content": None, "error": msg}
    else:
        call_kwargs = raw_kwargs

    filtered_kwargs = _accepted_kwargs(handler, call_kwargs, runtime_kwargs=raw_kwargs, ctx=ctx)

    try:
        result = await handler(**filtered_kwargs)
    except ToolRetryError:
        raise
    except Exception as exc:
        return {"content": None, "error": str(exc)}

    if isinstance(result, dict) and result.get("content") is None:
        result["content"] = ""
    return result
