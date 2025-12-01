"""Shared runner utilities for pydantic-ai models and tool registration."""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from pydantic_ai.messages import PartDeltaEvent, PartEndEvent  # type: ignore
from pydantic_ai.run import AgentRunResultEvent  # type: ignore
from isaac.agent.tools import TOOL_HANDLERS, run_tool


def register_tools(agent: Any) -> None:
    for name in TOOL_HANDLERS.keys():

        def _make_tool(fn_name: str):
            @agent.tool_plain(name=fn_name)  # type: ignore[misc]
            async def _wrapper(**kwargs: Any) -> Any:
                return await run_tool(fn_name, **kwargs)

            return _wrapper

        _make_tool(name)


async def run_with_runner(runner: Any, prompt_text: str, *, history: Any | None = None) -> str:
    run_method: Callable[[str], Any] | None = getattr(runner, "run", None)
    if not callable(run_method):
        return f"Echo: {prompt_text}"

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

        output = getattr(result, "output", None)
        if isinstance(output, str):
            return output
        if isinstance(result, str):
            return result
    except Exception as exc:  # pragma: no cover
        return f"Error: {exc}"

    return f"Echo: {prompt_text}"


async def stream_with_runner(
    runner: Any,
    prompt_text: str,
    on_text: Callable[[str], asyncio.Future | Any],
    cancel_event: asyncio.Event | None = None,
    *,
    history: Any | None = None,
) -> str | None:
    """Stream responses using the runner's streaming API."""
    cancel_event = cancel_event or asyncio.Event()
    stream_method: Callable[[str], Any] = getattr(runner, "run_stream_events")

    output_parts: list[str] = []
    event_iter = None
    if history is not None:
        try:
            event_iter = stream_method(prompt_text, messages=history)
        except TypeError:
            event_iter = None
    if event_iter is None:
        event_iter = stream_method(prompt_text)
    async for event in event_iter:
        if cancel_event.is_set():
            return None
        if isinstance(event, str):
            output_parts.append(event)
            await on_text(event)
            continue
        if isinstance(event, PartDeltaEvent):
            delta = getattr(event.delta, "content_delta", "")
            if delta:
                output_parts.append(delta)
                await on_text(delta)
        elif isinstance(event, PartEndEvent):
            part = getattr(event.part, "content", "")
            if part and not output_parts:
                output_parts.append(part)
                await on_text(part)
        elif isinstance(event, AgentRunResultEvent):
            result = getattr(event, "result", None)
            if result is not None and getattr(result, "output", None):
                full = getattr(result, "output")
                return str(full)

    return "".join(output_parts) or None
