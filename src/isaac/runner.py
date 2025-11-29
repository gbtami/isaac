"""Shared runner utilities for pydantic-ai models and tool registration."""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from isaac.tools import TOOL_HANDLERS, run_tool


def register_tools(agent: Any) -> None:
    for name in TOOL_HANDLERS.keys():

        def _make_tool(fn_name: str):
            @agent.tool_plain(name=fn_name)  # type: ignore[misc]
            async def _wrapper(**kwargs: Any) -> Any:
                return await run_tool(fn_name, **kwargs)

            return _wrapper

        _make_tool(name)


async def run_with_runner(runner: Any, prompt_text: str) -> str:
    run_method: Callable[[str], Any] | None = getattr(runner, "run", None)
    if not callable(run_method):
        return f"Echo: {prompt_text}"

    try:
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
