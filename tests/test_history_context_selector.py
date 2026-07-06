from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic_ai.messages import FunctionToolCallEvent, FunctionToolResultEvent, ToolCallPart, ToolReturnPart  # type: ignore

from isaac.agent.brain.history_utils import select_context_history
from isaac.agent.brain.prompt_handler import PromptHandler
from isaac.agent.brain.prompt_runner import PromptEnv, PromptRunner
from isaac.agent.brain.session_state import SessionState


def _noise(index: int) -> dict[str, str]:
    return {"role": "assistant", "content": f"old unrelated note {index}: " + ("noise " * 80)}


def test_context_selector_keeps_old_state_changing_tool_summary() -> None:
    history = [
        {
            "role": "assistant",
            "content": "Updated file src/critical.py [completed]\nDiff:\n+ important fix",
            "source": "tool_summary",
            "tool_name": "edit_file",
            "tool_kind": "edit",
        }
    ]
    history.extend(_noise(i) for i in range(40))
    history.extend({"role": "user", "content": f"recent user turn {i}"} for i in range(4))

    selected = select_context_history(
        history, current_prompt="continue", context_limit=4_000, recent_messages=4, max_messages=8
    )
    selected_text = "\n".join(str(msg.get("content") or "") for msg in selected)

    assert "src/critical.py" in selected_text
    assert "recent user turn 3" in selected_text
    assert "old unrelated note 0" not in selected_text


def test_context_selector_keeps_relevant_old_read_observation() -> None:
    history = [
        {
            "role": "assistant",
            "content": "Read file src/context_selector.py [completed]\nExcerpt:\nclass Selector: pass",
            "source": "tool_summary",
            "tool_name": "read_file",
            "tool_kind": "read",
        }
    ]
    history.extend(_noise(i) for i in range(40))
    history.extend({"role": "user", "content": f"recent user turn {i}"} for i in range(4))

    selected = select_context_history(
        history,
        current_prompt="Use src/context_selector.py for the next patch",
        context_limit=4_000,
        recent_messages=4,
        max_messages=8,
    )
    selected_text = "\n".join(str(msg.get("content") or "") for msg in selected)

    assert "src/context_selector.py" in selected_text
    assert "old unrelated note 0" not in selected_text


def test_context_selector_drops_irrelevant_old_read_observation_when_budgeted() -> None:
    history = [
        {
            "role": "assistant",
            "content": "Read file src/unrelated.py [completed]\nExcerpt:\nclass Unrelated: pass",
            "source": "tool_summary",
            "tool_name": "read_file",
            "tool_kind": "read",
        }
    ]
    history.extend(_noise(i) for i in range(40))
    history.extend({"role": "user", "content": f"recent user turn {i}"} for i in range(4))

    selected = select_context_history(
        history, current_prompt="continue the pytest fix", context_limit=4_000, recent_messages=4, max_messages=4
    )
    selected_text = "\n".join(str(msg.get("content") or "") for msg in selected)

    assert "src/unrelated.py" not in selected_text
    assert "recent user turn 3" in selected_text


def _make_env() -> PromptEnv:
    noop = AsyncMock()
    return PromptEnv(
        session_modes={},
        session_last_chunk={},
        send_message_chunk=noop,
        send_thought_chunk=noop,
        send_tool_start=noop,
        send_tool_finish=noop,
        send_plan_update=noop,
        send_notification=noop,
        send_protocol_update=noop,
        request_run_permission=AsyncMock(return_value=True),
        set_usage=lambda *_: None,
    )


@pytest.mark.asyncio
async def test_prompt_handler_uses_context_selector_for_old_tool_context(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_stream_with_runner(
        _runner: object,
        _prompt: str,
        *_: object,
        history: list[Any] | None = None,
        **__: object,
    ):
        captured["history"] = history
        return "ok", None

    monkeypatch.setattr("isaac.agent.brain.prompt_handler.stream_with_runner", fake_stream_with_runner)
    monkeypatch.setattr("isaac.agent.models.get_context_limit", lambda *_: 4_000)

    handler = PromptHandler(_make_env())
    history: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "content": "Updated file src/old_fix.py [completed]\nDiff:\n+ important",
            "source": "tool_summary",
            "tool_name": "edit_file",
            "tool_kind": "edit",
        }
    ]
    history.extend(_noise(i) for i in range(50))
    handler._sessions["s"] = SessionState(runner=object(), model_id="m", history=history)  # type: ignore[attr-defined]

    await handler.handle_prompt("s", "continue", asyncio.Event())

    sent_history = captured.get("history")
    assert isinstance(sent_history, list)
    assert any("src/old_fix.py" in str(msg.get("content") or "") for msg in sent_history if isinstance(msg, dict))


@pytest.mark.asyncio
async def test_prompt_runner_records_read_observations_as_tool_history() -> None:
    recorded: list[dict[str, Any]] = []
    noop = AsyncMock()
    runner = PromptRunner(
        PromptEnv(
            session_modes={},
            session_last_chunk={},
            send_message_chunk=noop,
            send_thought_chunk=noop,
            send_tool_start=noop,
            send_tool_finish=noop,
            send_plan_update=noop,
            send_notification=noop,
            send_protocol_update=noop,
            request_run_permission=AsyncMock(return_value=True),
            set_usage=lambda *_: None,
        )
    )
    handler = runner._build_runner_event_handler("s", record_history=lambda msg: recorded.append(dict(msg)))

    call = ToolCallPart(tool_name="read_file", args={"path": "src/foo.py", "start": 10, "lines": 2}, tool_call_id="tc1")
    result = ToolReturnPart(
        tool_name="read_file",
        content={"content": "line 10\nline 11\n", "sha256": "abc123", "error": None},
        tool_call_id="tc1",
    )

    await handler(FunctionToolCallEvent(part=call))
    await handler(FunctionToolResultEvent(part=result))

    assert recorded
    item = recorded[0]
    assert item["source"] == "tool_summary"
    assert item["tool_name"] == "read_file"
    assert item["tool_kind"] == "read"
    assert "src/foo.py" in item["content"]
    assert "line 10" in item["content"]
