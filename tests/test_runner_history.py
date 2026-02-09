from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic_ai import messages as ai_messages  # type: ignore
from pydantic_ai.messages import FunctionToolCallEvent, ToolCallPart  # type: ignore
from pydantic_ai.messages import PartDeltaEvent, TextPartDelta  # type: ignore
from pydantic_ai.run import AgentRunResultEvent  # type: ignore

from isaac.agent.runner import _final_text_correction, stream_with_runner
from isaac.agent.brain.prompt_runner import PromptEnv, PromptRunner
from isaac.agent.brain.tool_events import tool_history_summary


class _CapturingRunner:
    def __init__(self) -> None:
        self.captured_history: list[Any] | None = None

    def system_prompt(self) -> str:
        return "NEW_SYSTEM_PROMPT"

    def run_stream_events(self, *_: Any, message_history: Any = None, **__: Any) -> AsyncIterator[Any]:
        self.captured_history = message_history

        async def _gen() -> AsyncIterator[Any]:
            yield "ok"

        return _gen()


def _flatten_history(messages: list[Any]) -> list[dict[str, str]]:
    flattened: list[dict[str, str]] = []
    for message in messages:
        if isinstance(message, dict):
            flattened.append(message)
            continue
        if isinstance(message, ai_messages.ModelRequest):
            for part in message.parts:
                if isinstance(part, ai_messages.SystemPromptPart):
                    flattened.append({"role": "system", "content": str(part.content)})
                elif isinstance(part, ai_messages.UserPromptPart):
                    flattened.append({"role": "user", "content": str(part.content)})
        elif isinstance(message, ai_messages.ModelResponse):
            for part in message.parts:
                if isinstance(part, ai_messages.TextPart):
                    flattened.append({"role": "assistant", "content": str(part.content)})
    return flattened


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
    flattened = _flatten_history(runner.captured_history)
    assert flattened[0] == {"role": "system", "content": "OLD_SYSTEM_PROMPT"}
    assert any(msg.get("content") == "hi" for msg in flattened)


def test_tool_history_summary_uses_inputs() -> None:
    summary = tool_history_summary(
        "list_files",
        raw_output={"content": "ok"},
        status="completed",
        raw_input={"directory": "/tmp/demo"},
    )
    assert summary and "/tmp/demo" in summary


@pytest.mark.asyncio
async def test_tool_kinds_in_updates():
    captured: list[Any] = []

    async def _send_start(_session_id: str, event: Any) -> None:
        captured.append(event)

    noop = AsyncMock()
    env = PromptRunner(
        env=PromptEnv(
            session_modes={},
            session_last_chunk={},
            send_message_chunk=noop,
            send_thought_chunk=noop,
            send_tool_start=_send_start,
            send_tool_finish=noop,
            send_plan_update=noop,
            send_notification=noop,
            send_protocol_update=noop,
            request_run_permission=AsyncMock(return_value=True),
            set_usage=lambda *_: None,
        )
    )
    handler = env._build_runner_event_handler("s1", {}, None)
    event = FunctionToolCallEvent(
        part=ToolCallPart(tool_name="list_files", args={"directory": "/tmp"}, tool_call_id="tc1")
    )
    await handler(event)
    kinds = [getattr(u, "kind", None) for u in captured if u]
    assert "read" in kinds


@pytest.mark.asyncio
async def test_stream_with_runner_suppresses_duplicate_final_output() -> None:
    class _DuplicateStreamRunner:
        async def run_stream_events(self, *_: Any, **__: Any):
            async def _gen() -> AsyncIterator[Any]:
                # Duplicate chunk followed by the same final output.
                yield PartDeltaEvent(index=0, delta=TextPartDelta(content_delta="hello"))
                yield PartDeltaEvent(index=0, delta=TextPartDelta(content_delta="hello"))

                class _Result:
                    output = "hello"
                    usage = None

                    @staticmethod
                    def new_messages() -> list[Any]:
                        return []

                yield AgentRunResultEvent(result=_Result())

            return _gen()

    seen: list[str] = []

    async def on_text(text: str) -> None:
        if not seen or seen[-1] != text:
            seen.append(text)

    full, _usage = await stream_with_runner(
        _DuplicateStreamRunner(),
        "prompt",
        on_text=on_text,
        cancel_event=asyncio.Event(),
    )
    assert full == "hello"
    assert seen == ["hello"]


def test_final_text_correction_prefix_suffix_and_fallback() -> None:
    # Streamed prefix: append only the missing tail.
    assert _final_text_correction("Bud", "Budapest") == "apest"
    # Streamed suffix: rewrite the line with the full final text.
    assert _final_text_correction("apest", "Budapest") == "\r\x1b[2KBudapest"
    # Streamed middle substring: rewrite to final full text.
    assert _final_text_correction("uda", "Budapest") == "\r\x1b[2KBudapest"
    # Final already present in stream: do nothing.
    assert _final_text_correction("BudapestBudapest", "Budapest") is None
    # Generic single-line mismatch fallback: rewrite to final full text.
    assert _final_text_correction("xyz", "Budapest") == "\r\x1b[2KBudapest"
