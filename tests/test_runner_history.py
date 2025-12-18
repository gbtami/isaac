from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest
from pydantic_ai.messages import FunctionToolCallEvent, ToolCallPart  # type: ignore

from isaac.agent.runner import stream_with_runner
from isaac.agent.brain.strategy_runner import StrategyPromptRunner


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
    assert runner.captured_history[0] == {"role": "system", "content": "OLD_SYSTEM_PROMPT"}
    assert any(msg.get("content") == "hi" for msg in runner.captured_history)


def test_tool_history_summary_uses_inputs() -> None:
    summary = StrategyPromptRunner._tool_history_summary(
        "list_files",
        raw_output={"content": "ok"},
        status="completed",
        raw_input={"directory": "/tmp/demo"},
    )
    assert summary and "/tmp/demo" in summary


@pytest.mark.asyncio
async def test_tool_kinds_in_updates():
    captured: list[Any] = []

    async def _send(update, **_: Any):
        captured.append(update.update)

    class Env:
        session_modes: dict[str, str] = {}
        session_last_chunk: dict[str, str | None] = {}

        async def send_update(self, update, **_: Any):
            await _send(update)

        async def request_run_permission(self, *_args: Any, **_kwargs: Any) -> bool:
            return True

        def set_usage(self, *_args: Any, **_kwargs: Any) -> None:  # pragma: no cover - not used
            return None

    env = StrategyPromptRunner(env=Env())
    handler = env._build_runner_event_handler("s1", {}, {}, None)
    event = FunctionToolCallEvent(
        part=ToolCallPart(tool_name="list_files", args={"directory": "/tmp"}, tool_call_id="tc1")
    )
    await handler(event)
    kinds = [getattr(u, "kind", None) for u in captured if u]
    assert "read" in kinds
