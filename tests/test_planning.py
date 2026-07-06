from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from pydantic_ai.messages import FunctionToolCallEvent, FunctionToolResultEvent, ToolCallPart, ToolReturnPart  # type: ignore

from isaac.agent.brain.plan_progress import PlanProgress
from isaac.agent.brain.plan_schema import PlanStep, PlanSteps
from isaac.agent.brain.prompt_runner import PromptEnv, PromptRunner


@pytest.mark.asyncio
async def test_run_command_tool_start_includes_string_args():
    noop = AsyncMock()
    send_tool_start = AsyncMock()
    request_perm = AsyncMock(return_value=True)
    env = PromptEnv(
        session_modes={"s": "ask"},
        session_last_chunk={},
        send_message_chunk=noop,
        send_thought_chunk=noop,
        send_tool_start=send_tool_start,
        send_tool_finish=noop,
        send_plan_update=noop,
        send_notification=noop,
        send_protocol_update=noop,
        request_run_permission=request_perm,
        set_usage=lambda *_: None,
    )
    runner = PromptRunner(env)
    handler = runner._build_runner_event_handler(
        "s",
        plan_progress=None,
    )

    event = FunctionToolCallEvent(part=ToolCallPart(tool_name="run_command", args="echo hi"))
    await handler(event)

    request_perm.assert_not_awaited()

    send_tool_start.assert_awaited_once()
    call = send_tool_start.await_args
    event = call.args[1]
    assert event.raw_input.get("command") == "echo hi"


def test_plan_progress_marking_is_explicit_and_stable() -> None:
    progress = PlanProgress()
    plan = progress.set_plan(PlanSteps(entries=[PlanStep(content="Inspect"), PlanStep(content="Patch")]))

    assert [step.id for step in plan.entries] == ["step_1", "step_2"]
    assert progress.statuses == ["in_progress", "pending"]

    result = progress.mark(1, "completed", "inspection done")

    assert result.get("error") is None
    assert progress.statuses == ["completed", "in_progress"]


@pytest.mark.asyncio
async def test_successful_tool_result_does_not_auto_advance_plan() -> None:
    noop = AsyncMock()
    send_plan_update = AsyncMock()
    env = PromptEnv(
        session_modes={"s": "ask"},
        session_last_chunk={},
        send_message_chunk=noop,
        send_thought_chunk=noop,
        send_tool_start=noop,
        send_tool_finish=noop,
        send_plan_update=send_plan_update,
        send_notification=noop,
        send_protocol_update=noop,
        request_run_permission=AsyncMock(return_value=True),
        set_usage=lambda *_: None,
    )
    runner = PromptRunner(env)
    handler = runner._build_runner_event_handler(
        "s",
        plan_progress={"plan": PlanSteps(entries=[PlanStep(content="Real work")]), "idx": 0},
    )
    event = FunctionToolResultEvent(
        part=ToolReturnPart(tool_name="read_file", content={"content": "ok"}, tool_call_id="tc-read")
    )

    await handler(event)

    send_plan_update.assert_not_awaited()
