from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from isaac.agent.runner import stream_with_runner


class _CapturingRunner:
    def __init__(self) -> None:
        self.captured_history: list[dict[str, str]] | None = None

    def system_prompt(self) -> str:
        return "NEW_SYSTEM_PROMPT"

    def run_stream_events(self, *_: Any, message_history: Any = None, **__: Any) -> AsyncIterator[Any]:
        self.captured_history = message_history

        async def _gen() -> AsyncIterator[Any]:
            yield "ok"

        return _gen()


@pytest.mark.asyncio
async def test_stream_with_runner_replaces_system_prompt_in_history() -> None:
    runner = _CapturingRunner()

    async def on_text(_: str) -> None:
        return None

    await stream_with_runner(
        runner,
        "prompt",
        on_text,
        cancel_event=asyncio.Event(),
        history=[
            {"role": "system", "content": "OLD_SYSTEM_PROMPT"},
            {"role": "user", "content": "hi"},
        ],
    )

    assert runner.captured_history is not None
    assert runner.captured_history[0] == {"role": "system", "content": "NEW_SYSTEM_PROMPT"}
    assert all(msg.get("content") != "OLD_SYSTEM_PROMPT" for msg in runner.captured_history)
