"""Shared runner utilities for pydantic-ai models and tool registration."""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from pydantic_ai.messages import PartDeltaEvent, PartEndEvent  # type: ignore
from pydantic_ai.run import AgentRunResultEvent  # type: ignore
import logging

from isaac.agent.tools import TOOL_HANDLERS, run_tool, register_planning_tool
from isaac.agent.brain.planning_delegate import make_planning_tool


def register_tools(agent: Any, planning_agent: Any | None = None) -> None:
    logger = logging.getLogger("acp_server")

    for name in TOOL_HANDLERS.keys():

        def _make_tool(fn_name: str):
            @agent.tool_plain(name=fn_name)  # type: ignore[misc]
            async def _wrapper(**kwargs: Any) -> Any:
                logger.info("Pydantic tool invoked: %s args=%s", fn_name, sorted(kwargs))
                return await run_tool(fn_name, **kwargs)

            return _wrapper

        _make_tool(name)

    if planning_agent is not None:
        planning_handler = make_planning_tool(planning_agent)

        @agent.tool_plain(name="tool_generate_plan")  # type: ignore[misc]
        async def _planning_delegate(**kwargs: Any) -> Any:
            logger.info("Planning tool invoked via pydantic with args=%s", sorted(kwargs))
            return await planning_handler(**kwargs)

        register_planning_tool(planning_handler)


async def run_with_runner(
    runner: Any, prompt_text: str, *, history: Any | None = None
) -> tuple[str, Any | None]:
    run_method: Callable[[str], Any] | None = getattr(runner, "run", None)
    if not callable(run_method):
        return f"Echo: {prompt_text}", None

    usage = None
    try:
        result = None
        if history is not None:
            try:
                result = run_method(prompt_text, messages=history)
            except TypeError:
                result = None
        if result is None:
            result = run_method(prompt_text)
        if asyncio.iscoroutine(result):
            result = await result

        usage = getattr(result, "usage", None)
        output = getattr(result, "output", None)
        if isinstance(output, str):
            return output, usage
        if isinstance(result, str):
            return result, usage
    except Exception as exc:  # pragma: no cover
        return f"Provider error: {exc}", None

    return f"Echo: {prompt_text}", usage


async def stream_with_runner(
    runner: Any,
    prompt_text: str,
    on_text: Callable[[str], asyncio.Future | Any],
    cancel_event: asyncio.Event | None = None,
    *,
    history: Any | None = None,
    on_event: Callable[[Any], asyncio.Future | bool | None] | None = None,
) -> str | None:
    """Stream responses using the runner's streaming API.

    `on_event` lets callers react to tool call events (used to emit ACP tool updates).
    """
    stream_logger = logging.getLogger("acp_server")
    stream_logger.info("LLM stream start prompt_preview=%s", prompt_text[:200])
    cancel_event = cancel_event or asyncio.Event()
    stream_method: Callable[[str], Any] = getattr(runner, "run_stream_events")

    output_parts: list[str] = []
    usage = None
    try:
        event_iter = None
        if history is not None:
            try:
                event_iter = stream_method(prompt_text, messages=history)
            except TypeError:
                event_iter = None
        if event_iter is None:
            event_iter = stream_method(prompt_text)
        if asyncio.iscoroutine(event_iter):
            event_iter = await event_iter
        async for event in event_iter:
            if cancel_event.is_set():
                return None

            handled = False
            if on_event is not None:
                try:
                    handled = bool(await on_event(event))
                except TypeError:
                    # Support non-awaitable callbacks
                    handled = bool(on_event(event))

            if handled:
                continue

            if isinstance(event, str):
                output_parts.append(event)
                stream_logger.info("LLM text chunk=%s", event[:200].replace("\n", "\\n"))
                await on_text(event)
                continue
            if isinstance(event, PartDeltaEvent):
                delta = getattr(event.delta, "content_delta", "")
                if delta:
                    output_parts.append(delta)
                    stream_logger.info("LLM delta=%s", delta[:200].replace("\n", "\\n"))
                    await on_text(delta)
            elif isinstance(event, PartEndEvent):
                kind = getattr(event.part, "part_kind", "")
                if kind and kind not in {"text", "thinking"}:
                    logging.getLogger("acp_server").debug(
                        "Received PartEndEvent kind=%s content=%s", kind, getattr(event.part, "content", None)
                    )
                part = getattr(event.part, "content", "")
                if part and not output_parts:
                    output_parts.append(part)
                    stream_logger.info("LLM part_end=%s", str(part)[:200].replace("\n", "\\n"))
                    await on_text(part)
            elif isinstance(event, AgentRunResultEvent):
                result = getattr(event, "result", None)
                if result is not None and getattr(result, "output", None):
                    full = getattr(result, "output")
                    usage = getattr(result, "usage", None)
                    stream_logger.info("LLM final output=%s", str(full)[:200].replace("\n", "\\n"))
                    return str(full), usage
    except Exception as exc:  # pragma: no cover - provider errors
        stream_logger.warning("LLM stream error: %s", exc)
        return f"Provider error: {exc}", None

    return ("".join(output_parts) or None), usage
