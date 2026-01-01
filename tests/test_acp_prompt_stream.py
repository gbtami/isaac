from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from acp import text_block
from acp.schema import AgentMessageChunk, AgentPlanUpdate, ToolCallProgress, ToolCallStart
from pydantic_ai.messages import FunctionToolCallEvent, FunctionToolResultEvent, ToolCallPart  # type: ignore

from isaac.agent import ACPAgent
from isaac.agent.brain.plan_schema import PlanStep, PlanSteps


class _StreamRunner:
    def __init__(self, plan: PlanSteps) -> None:
        self._plan = plan

    def tool(self, *_: object, **__: object):
        def _decorator(func):
            return func

        return _decorator

    async def run_stream_events(self, prompt: str, message_history=None):
        _ = prompt, message_history

        async def _gen():
            part = ToolCallPart(tool_name="planner", args={"task": prompt}, tool_call_id="tc1")
            yield FunctionToolCallEvent(part=part)
            yield FunctionToolResultEvent(
                result=SimpleNamespace(tool_name="planner", content=self._plan, tool_call_id=part.tool_call_id)
            )
            yield "done"

        return _gen()


def _extract_updates(calls: list[object]) -> list[object]:
    updates: list[object] = []
    for call in calls:
        if hasattr(call, "kwargs") and "update" in call.kwargs:
            note = call.kwargs["update"]
        elif hasattr(call, "args") and call.args:
            note = call.args[0]
        else:
            continue
        updates.append(getattr(note, "update", note))
    return updates


def _index_of(updates: list[object], predicate) -> int:
    for idx, update in enumerate(updates):
        if predicate(update):
            return idx
    return -1


@pytest.mark.asyncio
async def test_acp_prompt_update_sequence(monkeypatch, tmp_path) -> None:
    conn = AsyncMock()
    conn.session_update = AsyncMock()
    conn.request_permission = AsyncMock(return_value=True)

    plan = PlanSteps(entries=[PlanStep(content="alpha"), PlanStep(content="beta")])
    runner = _StreamRunner(plan)

    from isaac.agent.brain import session_ops

    def _build(_model_id: str, _register: object, toolsets=None, **kwargs: object):
        _ = toolsets, kwargs
        return runner

    monkeypatch.setattr(session_ops, "create_subagent_for_model", _build)
    agent = ACPAgent(conn)

    session = await agent.new_session(cwd=str(tmp_path), mcp_servers=[])
    await agent.prompt(prompt=[text_block("do work")], session_id=session.session_id)

    updates = _extract_updates(conn.session_update.await_args_list)  # type: ignore[attr-defined]

    start_idx = _index_of(updates, lambda u: isinstance(u, ToolCallStart))
    finish_idx = _index_of(updates, lambda u: isinstance(u, ToolCallProgress) and u.status == "completed")
    plan_in_idx = _index_of(
        updates,
        lambda u: isinstance(u, AgentPlanUpdate)
        and any(getattr(entry, "status", "") == "in_progress" for entry in u.entries),
    )
    msg_idx = _index_of(
        updates,
        lambda u: isinstance(u, AgentMessageChunk) and getattr(u.content, "text", "").strip() == "done",
    )
    plan_done_idx = _index_of(
        updates,
        lambda u: isinstance(u, AgentPlanUpdate)
        and all(getattr(entry, "status", "") == "completed" for entry in u.entries),
    )

    assert start_idx != -1
    assert finish_idx != -1
    assert plan_in_idx != -1
    assert msg_idx != -1
    assert plan_done_idx != -1
    assert start_idx < finish_idx < plan_in_idx < msg_idx < plan_done_idx
