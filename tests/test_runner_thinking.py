from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest
from pydantic_ai.messages import PartDeltaEvent, PartEndEvent, ThinkingPart, ThinkingPartDelta

from isaac.agent.runner import stream_with_runner


class _FakeRunner:
    def __init__(self, events: list[object]) -> None:
        self._events = events

    def run_stream_events(self, *_: object, **__: object) -> AsyncIterator[object]:
        async def _gen() -> AsyncIterator[object]:
            for e in self._events:
                yield e

        return _gen()


@pytest.mark.asyncio
async def test_stream_with_runner_coalesces_thinking_deltas() -> None:
    events: list[object] = [
        PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="a")),
        PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="b")),
        PartEndEvent(index=0, part=ThinkingPart("ab")),
    ]
    runner = _FakeRunner(events)

    thoughts: list[str] = []

    async def on_text(_: str) -> None:
        raise AssertionError("unexpected text")

    async def on_thought(t: str) -> None:
        thoughts.append(t)

    await stream_with_runner(runner, "prompt", on_text=on_text, cancel_event=asyncio.Event(), on_thought=on_thought)

    assert thoughts == ["ab"]


@pytest.mark.asyncio
async def test_stream_with_runner_uses_delta_buffer_when_part_end_empty() -> None:
    events: list[object] = [
        PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="a")),
        PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="b")),
        PartEndEvent(index=0, part=ThinkingPart("")),
    ]
    runner = _FakeRunner(events)

    thoughts: list[str] = []

    async def on_text(_: str) -> None:
        raise AssertionError("unexpected text")

    async def on_thought(t: str) -> None:
        thoughts.append(t)

    await stream_with_runner(runner, "prompt", on_text=on_text, cancel_event=asyncio.Event(), on_thought=on_thought)

    assert thoughts == ["ab"]
