from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from pydantic_ai.messages import FunctionToolCallEvent, ToolCallPart  # type: ignore

from isaac.agent.brain.plan_parser import parse_plan_from_text
from isaac.agent.brain.prompt_runner import PromptEnv, PromptRunner


def test_plan_parser_handles_steps_list_line():
    update = parse_plan_from_text("steps=['first','second','third']")
    assert update
    contents = [getattr(e, "content", "") for e in update.entries]
    assert contents == ["first", "second", "third"]


@pytest.mark.asyncio
async def test_run_command_tool_start_includes_string_args():
    noop = AsyncMock()
    send_tool_start = AsyncMock()
    request_perm = AsyncMock(return_value=True)
    env = PromptEnv(
        session_modes={"s": "ask"},
        session_cwds={},
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
        run_command_ctx_tokens={},
        plan_progress=None,
    )

    event = FunctionToolCallEvent(part=ToolCallPart(tool_name="run_command", args="echo hi"))
    await handler(event)

    request_perm.assert_not_awaited()

    send_tool_start.assert_awaited_once()
    call = send_tool_start.await_args
    event = call.args[1]
    assert event.raw_input.get("command") == "echo hi"
