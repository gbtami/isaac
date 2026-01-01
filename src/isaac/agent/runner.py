"""Shared runner utilities for pydantic-ai models and tool registration."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import json
from typing import Any, Callable

from pydantic_ai import exceptions as ai_exc  # type: ignore
from pydantic_ai import messages as ai_messages  # type: ignore

from pydantic_ai.messages import (  # type: ignore
    PartDeltaEvent,
    PartEndEvent,
    TextPartDelta,
    ThinkingPartDelta,
)
from pydantic_ai.run import AgentRunResultEvent  # type: ignore
from isaac.log_utils import log_chunks_enabled, log_context as log_ctx, log_event

logger = logging.getLogger("isaac.agent.llm")

HISTORY_LOG_MAX = 8000


async def stream_with_runner(
    runner: Any,
    prompt_text: str,
    on_text: Callable[[str], asyncio.Future | Any],
    on_thought: Callable[[str], asyncio.Future | Any] | None = None,
    cancel_event: asyncio.Event | None = None,
    *,
    history: Any | None = None,
    on_event: Callable[[Any], asyncio.Future | bool | None] | None = None,
    store_messages: Callable[[Any], None] | None = None,
    log_context: str | None = None,
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

    def _sanitize_history(raw_history: Any) -> Any:
        """Drop invalid/empty history messages to avoid provider 400s."""

        cleaned = []
        if isinstance(raw_history, list):
            for msg in raw_history:
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role")
                content = msg.get("content")
                if not role or content is None:
                    continue
                content_str = str(content).strip()
                if content_str == "":
                    continue
                cleaned.append({"role": role, "content": content_str})
        return cleaned if cleaned else None

    def _convert_to_model_messages(raw_history: list[dict[str, str]] | None) -> list[Any] | None:
        """Pass through sanitized history; pydantic-ai accepts role/content dicts."""

        return raw_history

    def _model_messages_to_chat_history(raw_history: list[Any]) -> list[dict[str, str]]:
        """Convert pydantic-ai model messages into role/content chat history.

        This preserves user/assistant text while avoiding tool-call metadata
        that some providers reject when the history is truncated.
        """

        chat_history: list[dict[str, str]] = []
        for message in raw_history:
            if isinstance(message, dict):
                role = message.get("role")
                content = message.get("content")
                if role and content is not None:
                    chat_history.append({"role": str(role), "content": str(content)})
                continue
            if isinstance(message, ai_messages.ModelRequest):
                for part in getattr(message, "parts", ()) or ():
                    if isinstance(part, ai_messages.SystemPromptPart):
                        chat_history.append({"role": "system", "content": str(part.content)})
                    elif isinstance(part, ai_messages.UserPromptPart):
                        chat_history.append({"role": "user", "content": str(part.content)})
            elif isinstance(message, ai_messages.ModelResponse):
                for part in getattr(message, "parts", ()) or ():
                    if isinstance(part, ai_messages.TextPart):
                        chat_history.append({"role": "assistant", "content": str(part.content)})
        return chat_history

    event_iter = None
    try:
        thought_delta_buffer: list[str] = []
        if history is not None:
            history_preview = None
            try:
                history_preview = json.dumps(history)[:HISTORY_LOG_MAX]
            except Exception:
                history_preview = str(history)[:HISTORY_LOG_MAX]
            log_event(
                logger,
                "llm.history.raw",
                level=logging.DEBUG,
                history_preview=history_preview,
                history_len=len(history) if isinstance(history, list) else None,
            )
            if isinstance(history, list) and history:
                model_types = (ai_messages.ModelRequest, ai_messages.ModelResponse)
                if any(isinstance(msg, model_types) for msg in history):
                    chat_history = _model_messages_to_chat_history(history)
                    sanitized = _sanitize_history(chat_history)
                    safe_history = _convert_to_model_messages(sanitized)
                else:
                    sanitized = _sanitize_history(history)
                    safe_history = _convert_to_model_messages(sanitized)
            else:
                sanitized = _sanitize_history(history)
                safe_history = _convert_to_model_messages(sanitized)
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
            event_iter = runner.run_stream_events(prompt_text, message_history=safe_history)
        if event_iter is None:
            event_iter = runner.run_stream_events(prompt_text)
        if asyncio.iscoroutine(event_iter):
            event_iter = await event_iter
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
                        # stream_logger.info("LLM thinking delta=%s", str(delta)[:200].replace("\n", "\\n"))
                        # llm_logger.info("RECV thinking delta\n%s", delta)
                        thought_delta_buffer.append(str(delta))
                elif isinstance(event.delta, TextPartDelta):
                    delta = getattr(event.delta, "content_delta", "")
                    if delta:
                        output_parts.append(delta)
                        # stream_logger.info("LLM delta=%s", delta[:200].replace("\n", "\\n"))
                        # llm_logger.info("RECV delta\n%s", delta)
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
                    msgs = result.new_messages()
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
                    pass
                if not output_parts:
                    try:
                        await on_text(str(full))
                    except Exception:
                        pass
                return str(full), usage
    except (ai_exc.ModelRetry, ai_exc.UnexpectedModelBehavior) as exc:
        msg = str(exc)
        log_event(logger, "llm.stream.validation_failure", level=logging.WARNING, error=msg)
        friendly = "Model output failed validation; please retry or adjust the request."
        return friendly, usage
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
