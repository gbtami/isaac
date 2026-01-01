from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from acp.schema import (
    AgentMessageChunk,
    AgentPlanUpdate,
    AgentThoughtChunk,
    ToolCallProgress,
    ToolCallStart as ACPToolCallStart,
)

from isaac.agent.acp.prompt_env import ACPPromptEnvAdapter
from isaac.agent.brain.events import ToolCallFinish, ToolCallStart
from isaac.agent.brain.plan_schema import PlanStep, PlanSteps


@pytest.mark.asyncio
async def test_acp_prompt_env_emits_message_and_thought_chunks() -> None:
    send_update = AsyncMock()
    adapter = ACPPromptEnvAdapter(send_update=send_update)

    await adapter.send_message_chunk("s1", "hello")
    note = send_update.await_args.args[0]
    update = getattr(note, "update", note)
    assert isinstance(update, AgentMessageChunk)
    assert update.content.text == "hello"

    send_update.reset_mock()
    await adapter.send_thought_chunk("s1", "thinking")
    note = send_update.await_args.args[0]
    update = getattr(note, "update", note)
    assert isinstance(update, AgentThoughtChunk)
    assert update.content.text == "thinking"


@pytest.mark.asyncio
async def test_acp_prompt_env_tool_updates_include_diff() -> None:
    send_update = AsyncMock()
    adapter = ACPPromptEnvAdapter(send_update=send_update)

    start = ToolCallStart(
        tool_call_id="tc1",
        tool_name="edit_file",
        kind="edit",
        raw_input={"tool": "edit_file", "path": "demo.py"},
    )
    await adapter.send_tool_start("s1", start)
    note = send_update.await_args.args[0]
    update = getattr(note, "update", note)
    assert isinstance(update, ACPToolCallStart)
    assert update.status == "in_progress"
    assert update.raw_input.get("tool") == "edit_file"

    send_update.reset_mock()
    finish = ToolCallFinish(
        tool_call_id="tc1",
        tool_name="edit_file",
        status="completed",
        raw_output={"tool": "edit_file", "path": "demo.py", "new_text": "new", "old_text": "old"},
        old_text="old",
        new_text="new",
    )
    await adapter.send_tool_finish("s1", finish)
    note = send_update.await_args.args[0]
    update = getattr(note, "update", note)
    assert isinstance(update, ToolCallProgress)
    assert update.status == "completed"
    assert update.raw_output.get("path") == "demo.py"
    assert update.content
    assert update.content[0].type == "diff"


@pytest.mark.asyncio
async def test_acp_prompt_env_plan_updates_emit_progress() -> None:
    send_update = AsyncMock()
    adapter = ACPPromptEnvAdapter(send_update=send_update)

    steps = PlanSteps(entries=[PlanStep(content="first"), PlanStep(content="second")])
    await adapter.send_plan_update("s1", steps, active_index=0, status_all=None)
    note = send_update.await_args.args[0]
    update = getattr(note, "update", note)
    assert isinstance(update, AgentPlanUpdate)
    assert update.entries[0].status == "in_progress"
    assert update.entries[1].status == "pending"
