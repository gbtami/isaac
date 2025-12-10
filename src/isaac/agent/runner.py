"""Shared runner utilities for pydantic-ai models and tool registration."""

from __future__ import annotations

import asyncio
import inspect
import logging
import json
from typing import Any, Callable

from pydantic_ai import RunContext
from pydantic_ai import exceptions as ai_exc  # type: ignore

from pydantic_ai.messages import PartDeltaEvent, PartEndEvent  # type: ignore
from pydantic_ai.run import AgentRunResultEvent  # type: ignore

from isaac.agent.tools import TOOL_HANDLERS, run_tool


def register_tools(agent: Any) -> None:
    logger = logging.getLogger("acp_server")

    for name in TOOL_HANDLERS.keys():

        def _make_tool(fn_name: str):
            async def _wrapper(ctx: RunContext[Any] = None, **kwargs: Any) -> Any:
                logger.info("Pydantic tool invoked: %s args=%s", fn_name, sorted(kwargs))
                return await run_tool(fn_name, ctx=ctx, **kwargs)

            return agent.tool(name=fn_name)(_wrapper)  # type: ignore[misc]

        _make_tool(name)


async def stream_with_runner(
    runner: Any,
    prompt_text: str,
    on_text: Callable[[str], asyncio.Future | Any],
    cancel_event: asyncio.Event | None = None,
    *,
    history: Any | None = None,
    on_event: Callable[[Any], asyncio.Future | bool | None] | None = None,
) -> tuple[str | None, Any | None]:
    """Stream responses using the runner's streaming API.

    `on_event` lets callers react to tool call events (used to emit ACP tool updates).
    """
    stream_logger = logging.getLogger("acp_server")
    llm_logger = logging.getLogger("isaac.llm")
    stream_logger.info("LLM stream start prompt_preview=%s", prompt_text[:200])
    llm_logger.info("SEND prompt\n%s", prompt_text)
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

    try:
        event_iter = None
        if history is not None:
            try:
                llm_logger.info("HISTORY %s", json.dumps(history)[:2000])
            except Exception:
                llm_logger.info("HISTORY %s", str(history)[:2000])
            safe_history = _sanitize_history(history)
            event_iter = runner.run_stream_events(prompt_text, message_history=safe_history)
        if event_iter is None:
            event_iter = runner.run_stream_events(prompt_text)
        if asyncio.iscoroutine(event_iter):
            event_iter = await event_iter
        async for event in event_iter:
            if cancel_event.is_set():
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
                stream_logger.info("LLM text chunk=%s", event[:200].replace("\n", "\\n"))
                llm_logger.info("RECV text chunk\n%s", event)
                await on_text(event)
                continue
            if isinstance(event, PartDeltaEvent):
                delta = getattr(event.delta, "content_delta", "")
                if delta:
                    output_parts.append(delta)
                    stream_logger.info("LLM delta=%s", delta[:200].replace("\n", "\\n"))
                    llm_logger.info("RECV delta\n%s", delta)
                    await on_text(delta)
            elif isinstance(event, PartEndEvent):
                kind = getattr(event.part, "part_kind", "")
                if kind and kind not in {"text", "thinking"}:
                    logging.getLogger("acp_server").debug(
                        "Received PartEndEvent kind=%s content=%s",
                        kind,
                        getattr(event.part, "content", None),
                    )
                part = getattr(event.part, "content", "")
                if part and not output_parts:
                    output_parts.append(part)
                    stream_logger.info("LLM part_end=%s", str(part)[:200].replace("\n", "\\n"))
                    llm_logger.info("RECV part_end\n%s", part)
                    await on_text(part)
            elif isinstance(event, AgentRunResultEvent):
                result = getattr(event, "result", None)
                if result is not None and getattr(result, "output", None):
                    full = getattr(result, "output")
                    usage = getattr(result, "usage", None)
                    stream_logger.info("LLM final output=%s", str(full)[:200].replace("\n", "\\n"))
                    llm_logger.info("RECV final output\n%s", full)
                    return str(full), usage
    except (ai_exc.ModelRetry, ai_exc.UnexpectedModelBehavior) as exc:
        msg = str(exc)
        stream_logger.warning("LLM validation failure: %s", msg)
        llm_logger.warning("LLM validation failure: %s", msg)
        friendly = "Model output failed validation; please retry or adjust the request."
        return friendly, usage
    except Exception as exc:  # pragma: no cover - provider errors
        msg = str(exc)
        stream_logger.warning("LLM stream error: %s", msg)
        llm_logger.warning("LLM stream error: %s", msg)
        return f"Provider error: {msg}", None

    return ("".join(output_parts) or None), usage
