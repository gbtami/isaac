from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic_ai.messages import FunctionToolCallEvent, FunctionToolResultEvent, ToolCallPart, ToolReturnPart  # type: ignore

from isaac.agent.brain.memory import (
    CodingMemory,
    CodingMemoryEvent,
    memory_events_from_tool_result,
    select_memory_events,
    selected_memory_context,
)
from isaac.agent.brain.prompt_handler import PromptHandler
from isaac.agent.brain.prompt_runner import PromptEnv, PromptRunner
from isaac.agent.brain.recent_files import record_recent_file
from isaac.agent.brain.session_state import SessionState


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


def test_memory_events_extract_structured_read_file_observation() -> None:
    events = memory_events_from_tool_result(
        "read_file",
        {"content": "class Brain: pass\n", "sha256": "abc123", "error": None},
        "completed",
        raw_input={"path": "src/isaac/brain.py", "start": 3, "lines": 5},
    )

    assert len(events) == 1
    event = events[0]
    assert event.kind == "observation"
    assert event.tool_name == "read_file"
    assert event.paths == ["src/isaac/brain.py"]
    assert event.sha256 == "abc123"
    assert event.metadata["start"] == 3
    assert "class Brain" in event.summary


def test_memory_events_expand_coding_delegate_artifacts() -> None:
    events = memory_events_from_tool_result(
        "coding",
        {
            "content": {
                "summary": "Implemented structured delegate artifact propagation.",
                "files": [
                    {
                        "path": "src/isaac/agent/brain/memory.py",
                        "action": "changed",
                        "summary": "Added delegate artifact memory extraction.",
                        "intent": "Keep parent memory file-aware.",
                    }
                ],
                "tests": ["uv run pytest tests/test_coding_memory.py"],
                "risks": ["Delegate file actions may be approximate."],
                "followups": ["Run the full test suite."],
            },
            "delegate_tool": "coding",
            "delegate_session_id": "ds",
            "delegate_run_id": "dr",
            "error": None,
        },
        "completed",
        raw_input={"task": "implement P3"},
    )

    kinds = [event.kind for event in events]
    assert kinds == ["delegate", "edit", "test", "risk", "followup"]
    file_event = events[1]
    assert file_event.paths == ["src/isaac/agent/brain/memory.py"]
    assert file_event.metadata["artifact_type"] == "file"
    assert file_event.metadata["delegate_session_id"] == "ds"
    assert "Keep parent memory file-aware" in file_event.summary


def test_memory_events_expand_review_delegate_findings() -> None:
    events = memory_events_from_tool_result(
        "review",
        {
            "content": {
                "summary": "Found one correctness issue.",
                "findings": [
                    {
                        "severity": "high",
                        "description": "Session cwd is ignored for model tools.",
                        "file": "src/isaac/agent/tools/registration.py",
                        "line": 42,
                        "suggestion": "Pass SessionToolDeps into wrappers.",
                    }
                ],
                "tests": ["Add cwd regression test."],
            },
            "delegate_tool": "review",
            "error": None,
        },
        "completed",
        raw_input={"task": "review P0"},
    )

    finding = next(event for event in events if event.kind == "finding")
    assert finding.status == "open"
    assert finding.paths == ["src/isaac/agent/tools/registration.py"]
    assert finding.metadata["severity"] == "high"
    assert "Pass SessionToolDeps" in finding.summary


def test_memory_selection_keeps_relevant_old_observation_and_important_edits() -> None:
    memory = CodingMemory()
    memory.extend(
        [
            CodingMemoryEvent(
                kind="observation", summary="Read file src/unrelated.py noise", paths=["src/unrelated.py"]
            ),
            CodingMemoryEvent(
                kind="observation",
                summary="Read file src/context_selector.py class Selector",
                paths=["src/context_selector.py"],
            ),
            CodingMemoryEvent(
                kind="edit", summary="Updated file src/old_fix.py with important patch", paths=["src/old_fix.py"]
            ),
        ]
    )

    selected = select_memory_events(
        memory.events,
        current_prompt="continue the src/context_selector.py refactor",
        context_limit=4_000,
        max_events=2,
    )
    text = "\n".join(event.summary for event in selected)

    assert "src/context_selector.py" in text
    assert "src/old_fix.py" in text
    assert "src/unrelated.py" not in text


def test_selected_memory_context_is_bounded_and_actionable() -> None:
    memory = CodingMemory()
    memory.append(
        CodingMemoryEvent(
            kind="command",
            summary="Ran command: uv run pytest [failed]\nStderr:\nE assert 1 == 2",
            status="failed",
            command="uv run pytest",
        )
    )

    context = selected_memory_context(memory, current_prompt="fix pytest", context_limit=8_000)

    assert context is not None
    assert "Session coding memory" in context
    assert "uv run pytest" in context
    assert "Re-read files before exact edits" in context


@pytest.mark.asyncio
async def test_prompt_runner_records_structured_memory_events() -> None:
    recorded: list[CodingMemoryEvent] = []
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
    handler = runner._build_runner_event_handler("s", record_memory=recorded.append)

    call = ToolCallPart(tool_name="run_command", args={"command": "uv run pytest"}, tool_call_id="tc1")
    result = ToolReturnPart(
        tool_name="run_command",
        content={"content": "", "error": "test failed", "returncode": 1},
        tool_call_id="tc1",
    )

    await handler(FunctionToolCallEvent(part=call))
    await handler(FunctionToolResultEvent(part=result))

    assert len(recorded) == 1
    event = recorded[0]
    assert event.kind == "command"
    assert event.status == "failed"
    assert event.command == "uv run pytest"
    assert "test failed" in event.summary


@pytest.mark.asyncio
async def test_prompt_runner_records_delegate_artifact_memory_events() -> None:
    recorded: list[CodingMemoryEvent] = []
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
    handler = runner._build_runner_event_handler("s", record_memory=recorded.append)

    call = ToolCallPart(tool_name="coding", args={"task": "edit memory"}, tool_call_id="tc1")
    result = ToolReturnPart(
        tool_name="coding",
        content={
            "content": {
                "summary": "Edited memory handling.",
                "files": [
                    {
                        "path": "src/isaac/agent/brain/memory.py",
                        "action": "changed",
                        "summary": "Expanded delegate artifacts into events.",
                    }
                ],
                "tests": [],
                "risks": [],
                "followups": [],
            },
            "error": None,
        },
        tool_call_id="tc1",
    )

    await handler(FunctionToolCallEvent(part=call))
    await handler(FunctionToolResultEvent(part=result))

    assert [event.kind for event in recorded] == ["delegate", "edit"]
    assert recorded[1].paths == ["src/isaac/agent/brain/memory.py"]


def test_recent_files_tracks_coding_delegate_file_artifacts() -> None:
    recent_files: list[str] = []
    result = ToolReturnPart(
        tool_name="coding",
        content={
            "content": {
                "summary": "Changed files.",
                "files": [
                    {"path": "src/a.py", "action": "changed", "summary": "Updated A."},
                    {"path": "src/b.py", "action": "created", "summary": "Added B."},
                ],
            },
            "error": None,
        },
        tool_call_id="tc1",
    )

    record_recent_file(recent_files, FunctionToolResultEvent(part=result), 5)

    assert recent_files == ["src/a.py", "src/b.py"]


@pytest.mark.asyncio
async def test_prompt_handler_passes_coding_memory_as_pydantic_capability(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_stream_with_runner(
        _runner: object,
        _prompt: str,
        *_: object,
        capabilities: list[Any] | None = None,
        **__: object,
    ):
        captured["capabilities"] = capabilities
        return "ok", None

    monkeypatch.setattr("isaac.agent.brain.prompt_handler.stream_with_runner", fake_stream_with_runner)
    monkeypatch.setattr("isaac.agent.models.get_context_limit", lambda *_: 8_000)

    memory = CodingMemory()
    memory.append(
        CodingMemoryEvent(
            kind="edit",
            summary="Updated file src/memory.py with durable event support",
            paths=["src/memory.py"],
        )
    )
    handler = PromptHandler(_make_env())
    handler._sessions["s"] = SessionState(runner=object(), model_id="m", coding_memory=memory)  # type: ignore[attr-defined]

    await handler.handle_prompt("s", "continue memory work", asyncio.Event())

    capabilities = captured.get("capabilities")
    assert isinstance(capabilities, list)
    memory_capabilities = [cap for cap in capabilities if getattr(cap, "id", None) == "isaac-coding-memory"]
    assert memory_capabilities
    instructions = memory_capabilities[0].get_instructions()
    assert instructions is not None
    assert "src/memory.py" in str(instructions)


def test_coding_memory_round_trips_through_snapshot() -> None:
    memory = CodingMemory()
    memory.append(CodingMemoryEvent(kind="edit", summary="Updated src/foo.py", paths=["src/foo.py"]))
    snapshot = memory.model_dump_jsonable()

    restored = CodingMemory.from_snapshot(snapshot)

    assert len(restored.events) == 1
    assert restored.events[0].paths == ["src/foo.py"]
