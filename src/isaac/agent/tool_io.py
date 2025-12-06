"""Utilities for handling tool I/O safely."""

from __future__ import annotations

import asyncio
from typing import Any


def truncate_text(value: str, limit: int) -> tuple[str, bool]:
    """Clamp tool output strings to avoid exceeding ACP stream line limits."""

    if len(value) <= limit:
        return value, False
    suffix = "\n[truncated]"
    keep = max(0, limit - len(suffix))
    return f"{value[:keep]}{suffix}", True


def truncate_tool_output(result: dict[str, Any], limit: int) -> tuple[dict[str, Any], bool]:
    """Return a copy of tool output with oversized text fields truncated."""

    truncated = False
    trimmed = dict(result)
    for key in ("content", "error"):
        val = trimmed.get(key)
        if isinstance(val, str):
            new_val, did_truncate = truncate_text(val, limit)
            truncated = truncated or did_truncate
            trimmed[key] = new_val
    return trimmed, truncated


async def await_with_cancel(coro: Any, cancel_event: asyncio.Event) -> Any | None:
    """Await a coroutine while honoring cancellation from an asyncio.Event."""

    wait_task = asyncio.create_task(cancel_event.wait())
    main_task = asyncio.create_task(coro)
    done, _pending = await asyncio.wait(
        {main_task, wait_task},
        return_when=asyncio.FIRST_COMPLETED,
    )
    if wait_task in done and cancel_event.is_set():
        main_task.cancel()
        return None
    return main_task.result() if main_task in done else None
