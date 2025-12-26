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
async def test_run_command_permission_includes_string_args():
    send_update = AsyncMock()
    request_perm = AsyncMock(return_value=True)
    env = PromptEnv(
        session_modes={"s": "ask"},
        session_last_chunk={},
        send_update=send_update,
        request_run_permission=request_perm,
        set_usage=lambda *_: None,
    )
    runner = PromptRunner(env)
    handler = runner._build_runner_event_handler(
        "s",
        tool_trackers={},
        run_command_ctx_tokens={},
        plan_progress=None,
    )

    event = FunctionToolCallEvent(part=ToolCallPart(tool_name="run_command", args="echo hi"))
    await handler(event)

    request_perm.assert_awaited_once()
    assert request_perm.await_args.kwargs["command"] == "echo hi"

    updates = [call.args[0] for call in send_update.await_args_list]  # type: ignore[attr-defined]
    assert updates
    start_update = updates[0].update
    assert getattr(start_update, "raw_input", {}).get("command") == "echo hi"
