"""Shared runner utilities for pydantic-ai models and tool registration."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
from typing import Any, Callable, Protocol, Sequence

import httpx
from pydantic_ai import DeferredToolRequests, DeferredToolResults, ToolApproved, ToolDenied  # type: ignore
from pydantic_ai import exceptions as ai_exc  # type: ignore
from pydantic_ai import messages as ai_messages  # type: ignore

from pydantic_ai.messages import (  # type: ignore
    PartDeltaEvent,
    PartEndEvent,
    TextPartDelta,
    ThinkingPartDelta,
)
from pydantic_ai.run import AgentRunResultEvent  # type: ignore
from isaac.agent.history_types import ChatMessage, HistoryInput
from isaac.log_utils import log_chunks_enabled, log_context as log_ctx, log_event

logger = logging.getLogger("isaac.agent.llm")

HISTORY_LOG_MAX = 8000


class StreamRunner(Protocol):
    """Protocol for pydantic-ai runners that stream events."""

    def run_stream_events(
        self,
        prompt_text: str | None,
        *,
        message_history: Sequence[ai_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
    ) -> Any: ...


async def stream_with_runner(
    runner: StreamRunner,
    prompt_text: str,
    on_text: Callable[[str], asyncio.Future | Any],
    on_thought: Callable[[str], asyncio.Future | Any] | None = None,
    cancel_event: asyncio.Event | None = None,
    *,
    history: HistoryInput | None = None,
    on_event: Callable[[Any], asyncio.Future | bool | None] | None = None,
    store_messages: Callable[[Any], None] | None = None,
    log_context: str | None = None,
    request_tool_approval: Callable[[str, str, dict[str, Any]], asyncio.Future | bool] | None = None,
    usage_limits: Any | None = None,
    metadata: dict[str, Any] | None = None,
) -> tuple[str | None, Any | None]:
    """Stream responses using the runner's streaming API.

    `on_event` lets callers react to tool call events (used to emit ACP tool updates).
    """
    with log_ctx(llm_context=log_context):
        log_event(
            logger,
            "llm.stream.start",
            prompt_preview=prompt_text[:200].replace("\n", "\\n"),
            has_history=history is not None,
        )
        log_event(logger, "llm.prompt.send", level=logging.DEBUG, prompt=prompt_text)
    cancel_event = cancel_event or asyncio.Event()

    output_parts: list[str] = []
    usage = None

    def _convert_to_model_messages(
        raw_history: Sequence[ChatMessage] | None,
    ) -> list[ai_messages.ModelMessage] | None:
        """Convert chat history dicts into pydantic-ai ModelMessage instances."""

        if not raw_history:
            return None
        messages: list[ai_messages.ModelMessage] = []
        for msg in raw_history:
            role = msg.get("role")
            content = msg.get("content")
            if content is None:
                continue
            text = str(content)
            if role == "assistant":
                messages.append(ai_messages.ModelResponse(parts=[ai_messages.TextPart(content=text)]))
            elif role == "system":
                messages.append(ai_messages.ModelRequest(parts=[ai_messages.SystemPromptPart(content=text)]))
            else:
                messages.append(ai_messages.ModelRequest(parts=[ai_messages.UserPromptPart(content=text)]))
        return messages or None

    async def _resolve_deferred_tool_results(requests: DeferredToolRequests) -> DeferredToolResults:
        approvals: dict[str, ToolApproved | ToolDenied] = {}
        for approval in requests.approvals:
            tool_name = str(getattr(approval, "tool_name", "") or "")
            tool_call_id = str(getattr(approval, "tool_call_id", "") or "")
            raw_args = getattr(approval, "args", None)
            args = dict(raw_args) if isinstance(raw_args, dict) else {}
            allowed = False
            if request_tool_approval is None:
                allowed = False
            elif asyncio.iscoroutinefunction(request_tool_approval):
                allowed = bool(await request_tool_approval(tool_call_id, tool_name, args))
            else:
                maybe_allowed = request_tool_approval(tool_call_id, tool_name, args)
                if inspect.isawaitable(maybe_allowed):
                    allowed = bool(await maybe_allowed)
                else:
                    allowed = bool(maybe_allowed)
            approvals[tool_call_id] = ToolApproved() if allowed else ToolDenied("permission denied")
        return DeferredToolResults(approvals=approvals, metadata=requests.metadata or {})

    def _run_stream_events(
        prompt: str | None,
        *,
        message_history: list[ai_messages.ModelMessage] | None,
        deferred_results: DeferredToolResults | None,
    ) -> Any:
        kwargs: dict[str, Any] = {}
        if message_history is not None:
            kwargs["message_history"] = message_history
        if deferred_results is not None:
            kwargs["deferred_tool_results"] = deferred_results
        if usage_limits is not None:
            kwargs["usage_limits"] = usage_limits
        if metadata is not None:
            kwargs["metadata"] = metadata
        try:
            return runner.run_stream_events(prompt, **kwargs)
        except TypeError:
            kwargs.pop("metadata", None)
            kwargs.pop("usage_limits", None)
            kwargs.pop("deferred_tool_results", None)
            return runner.run_stream_events(prompt, **kwargs)

    event_iter = None
    safe_history: list[ai_messages.ModelMessage] | None = None
    deferred_tool_results: DeferredToolResults | None = None
    next_prompt: str | None = prompt_text
    history_list = list(history) if history is not None else None
    if history_list is not None:
        history_preview = None
        try:
            history_preview = json.dumps(history_list)[:HISTORY_LOG_MAX]
        except Exception:
            history_preview = str(history_list)[:HISTORY_LOG_MAX]
        log_event(
            logger,
            "llm.history.raw",
            level=logging.DEBUG,
            history_preview=history_preview,
            history_len=len(history_list),
        )
        model_types = (ai_messages.ModelRequest, ai_messages.ModelResponse)
        if history_list and any(isinstance(msg, model_types) for msg in history_list):
            safe_history = list(history_list)
        else:
            safe_history = _convert_to_model_messages(history_list)
        sanitized_preview = None
        try:
            sanitized_preview = json.dumps(safe_history)[:HISTORY_LOG_MAX]
        except Exception:
            sanitized_preview = str(safe_history)[:HISTORY_LOG_MAX]
        log_event(
            logger,
            "llm.history.sanitized",
            level=logging.DEBUG,
            history_preview=sanitized_preview,
            history_len=len(safe_history) if isinstance(safe_history, list) else None,
        )
    try:
        thought_delta_buffer: list[str] = []
        while True:
            event_iter = None
            event_iter = _run_stream_events(
                next_prompt,
                message_history=safe_history,
                deferred_results=deferred_tool_results,
            )
            next_prompt = None
            deferred_tool_results = None
            if asyncio.iscoroutine(event_iter):
                event_iter = await event_iter

            continue_after_approval = False
            async for event in event_iter:
                if cancel_event.is_set():
                    closer = getattr(event_iter, "aclose", None)
                    if callable(closer):
                        with contextlib.suppress(Exception):
                            await closer()
                    return None, usage

                handled = False
                if on_event is not None:
                    if asyncio.iscoroutinefunction(on_event):
                        handled = bool(await on_event(event))
                    else:
                        maybe = on_event(event)
                        if inspect.isawaitable(maybe):
                            handled = bool(await maybe)
                        else:
                            handled = bool(maybe)

                if handled:
                    continue

                if isinstance(event, str):
                    output_parts.append(event)
                    if log_chunks_enabled():
                        log_event(
                            logger,
                            "llm.stream.text",
                            level=logging.DEBUG,
                            chunk_preview=event[:200].replace("\n", "\\n"),
                        )
                        log_event(logger, "llm.stream.text.raw", level=logging.DEBUG, chunk=event)
                    await on_text(event)
                    continue
                if isinstance(event, PartDeltaEvent):
                    if isinstance(event.delta, ThinkingPartDelta):
                        delta = getattr(event.delta, "content_delta", None)
                        if delta:
                            thought_delta_buffer.append(str(delta))
                    elif isinstance(event.delta, TextPartDelta):
                        delta = getattr(event.delta, "content_delta", "")
                        if delta:
                            output_parts.append(delta)
                            await on_text(delta)
                elif isinstance(event, PartEndEvent):
                    kind = getattr(event.part, "part_kind", "")
                    if kind and kind not in {"text", "thinking"}:
                        if log_chunks_enabled():
                            log_event(
                                logger,
                                "llm.stream.part_end.unknown",
                                level=logging.DEBUG,
                                kind=kind,
                                content=getattr(event.part, "content", None),
                            )
                    part = getattr(event.part, "content", "")
                    if kind == "thinking" and on_thought is not None:
                        final_thought = str(part) if part else "".join(thought_delta_buffer)
                        thought_delta_buffer.clear()
                        if final_thought:
                            if log_chunks_enabled():
                                log_event(
                                    logger,
                                    "llm.stream.thinking",
                                    level=logging.DEBUG,
                                    thought_preview=final_thought[:200].replace("\n", "\\n"),
                                )
                                log_event(
                                    logger,
                                    "llm.stream.thinking.raw",
                                    level=logging.DEBUG,
                                    thought=final_thought,
                                )
                            await on_thought(final_thought)
                    elif part and not output_parts:
                        output_parts.append(part)
                        if log_chunks_enabled():
                            log_event(
                                logger,
                                "llm.stream.part_end",
                                level=logging.DEBUG,
                                part_preview=str(part)[:200].replace("\n", "\\n"),
                            )
                            log_event(logger, "llm.stream.part_end.raw", level=logging.DEBUG, part=part)
                        await on_text(part)
                elif isinstance(event, AgentRunResultEvent):
                    result = event.result
                    full = result.output
                    usage = getattr(result, "usage", None)
                    log_event(
                        logger,
                        "llm.stream.final",
                        output_preview=str(full)[:200].replace("\n", "\\n"),
                    )
                    log_event(logger, "llm.stream.final.raw", level=logging.DEBUG, output=full)
                    try:
                        msgs = list(result.new_messages())
                        log_event(
                            logger,
                            "llm.model_messages",
                            level=logging.DEBUG,
                            messages_preview=str(msgs)[:HISTORY_LOG_MAX],
                        )
                        if store_messages is not None:
                            with contextlib.suppress(Exception):
                                store_messages(msgs)
                    except Exception:
                        msgs = []
                    if isinstance(full, DeferredToolRequests):
                        deferred_tool_results = await _resolve_deferred_tool_results(full)
                        if safe_history:
                            safe_history = [*safe_history, *msgs]
                        elif msgs:
                            safe_history = msgs
                        continue_after_approval = True
                        break
                    try:
                        full_text = "" if full is None else str(full)
                        if not output_parts:
                            await on_text(full_text)
                    except Exception:
                        pass
                    return str(full), usage

            if continue_after_approval:
                continue
            break
    except (ai_exc.ModelRetry, ai_exc.UnexpectedModelBehavior) as exc:
        msg = str(exc)
        log_event(logger, "llm.stream.validation_failure", level=logging.WARNING, error=msg)
        friendly = "Model output failed validation; please retry or adjust the request."
        return friendly, usage
    except httpx.TimeoutException as exc:
        msg = str(exc) or "Request timed out."
        log_event(logger, "llm.stream.timeout", level=logging.WARNING, error=msg)
        return f"Provider timeout: {msg}", None
    except Exception as exc:  # pragma: no cover - provider errors
        msg = str(exc)
        log_event(logger, "llm.stream.error", level=logging.WARNING, error=msg)
        return f"Provider error: {msg}", None
    finally:
        if event_iter is not None:
            closer = getattr(event_iter, "aclose", None)
            if callable(closer):
                with contextlib.suppress(Exception):
                    await closer()

    if thought_delta_buffer and on_thought is not None:
        final_thought = "".join(thought_delta_buffer)
        if final_thought:
            with contextlib.suppress(Exception):
                await on_thought(final_thought)

    return ("".join(output_parts) or None), usage
