from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from isaac.agent.runner import stream_with_runner


class _AsyncEventContext:
    def __init__(self, events: list[object]) -> None:
        self.events = events
        self.entered = False
        self.exited = False

    async def __aenter__(self) -> AsyncIterator[object]:
        self.entered = True

        async def _gen() -> AsyncIterator[object]:
            for event in self.events:
                yield event

        return _gen()

    async def __aexit__(self, *_: object) -> None:
        self.exited = True


class _ContextManagerRunner:
    def __init__(self, events: list[object]) -> None:
        self.context = _AsyncEventContext(events)
        self.kwargs: dict[str, Any] | None = None

    def run_stream_events(self, prompt_text: str | None, **kwargs: Any) -> _AsyncEventContext:
        self.kwargs = {"prompt_text": prompt_text, **kwargs}
        return self.context


@pytest.mark.asyncio
async def test_stream_with_runner_accepts_pydantic_ai_v2_context_manager() -> None:
    runner = _ContextManagerRunner(["hello"])
    seen: list[str] = []

    async def on_text(text: str) -> None:
        seen.append(text)

    full, usage = await stream_with_runner(runner, "prompt", on_text, cancel_event=asyncio.Event())

    assert full == "hello"
    assert usage is None
    assert seen == ["hello"]
    assert runner.context.entered is True
    assert runner.context.exited is True


@pytest.mark.asyncio
async def test_stream_with_runner_passes_approval_as_per_run_capability() -> None:
    runner = _ContextManagerRunner(["ok"])

    async def on_text(_: str) -> None:
        return None

    async def approve(_call_id: str, _tool_name: str, _args: dict[str, Any]) -> bool:
        return True

    await stream_with_runner(
        runner,
        "prompt",
        on_text,
        cancel_event=asyncio.Event(),
        request_tool_approval=approve,
    )

    assert runner.kwargs is not None
    capabilities = runner.kwargs.get("capabilities")
    assert capabilities
    assert any(type(capability).__name__ == "ACPPermissionCapability" for capability in capabilities)
