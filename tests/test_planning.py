from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from acp import PromptRequest, text_block
from acp.schema import AgentPlanUpdate
from pydantic_ai import Agent as PydanticAgent  # type: ignore
from pydantic_ai.models.test import TestModel  # type: ignore

from isaac.agent.brain.planning_delegate import build_planning_agent
from isaac.agent.runner import register_tools
from tests.utils import make_function_agent


@pytest.mark.asyncio
async def test_plan_updates():
    conn = AsyncMock()
    agent = make_function_agent(conn)

    session_id = "plan-session"
    response = await agent.prompt(
        PromptRequest(sessionId=session_id, prompt=[text_block("plan:step one;step two")])
    )

    conn.sessionUpdate.assert_called_once()
    notification = conn.sessionUpdate.call_args[0][0]
    assert notification.sessionId == session_id
    assert hasattr(notification.update, "entries")
    assert len(notification.update.entries) == 2
    assert response.stopReason == "end_turn"


@pytest.mark.asyncio
async def test_plan_tool_emits_agent_plan():
    conn = AsyncMock()
    agent = make_function_agent(conn)

    session_id = "plan-tool-session"
    response = await agent.prompt(
        PromptRequest(sessionId=session_id, prompt=[text_block("tool:plan ship a release")])
    )

    assert response.stopReason == "end_turn"
    updates = [call.args[0].update for call in conn.sessionUpdate.await_args_list]  # type: ignore[attr-defined]
    assert any(getattr(u, "entries", None) for u in updates)


@pytest.mark.asyncio
@pytest.mark.parametrize("plan_mode", ["structured", "text"])
async def test_planning_tool_emits_plan_update(plan_mode: str):
    conn = AsyncMock()
    # Main runner always calls the planning tool.
    runner = PydanticAgent(TestModel(call_tools=["tool_generate_plan"]))

    if plan_mode == "structured":
        planning_runner = build_planning_agent(
            TestModel(call_tools=[], custom_output_args={"steps": ["alpha", "beta"]})
        )
    else:
        planning_runner = build_planning_agent(TestModel(call_tools=[]))

        class FakePlanResult:
            def __init__(self):
                self.output = None
                self.response = SimpleNamespace(parts=[SimpleNamespace(content="Plan:\n- alpha\n- beta")])

        planning_runner.run = AsyncMock(return_value=FakePlanResult())

    register_tools(runner, planning_agent=planning_runner)
    agent = make_function_agent(conn)
    # swap runner and planning agent into ACPAgent instance
    agent._ai_runner = runner  # type: ignore[attr-defined]

    session_id = f"plan-session-{plan_mode}"
    await agent.prompt(PromptRequest(sessionId=session_id, prompt=[text_block("need a plan")]))

    plan_updates = [
        call_args[0][0].update
        for call_args in conn.sessionUpdate.call_args_list
        if isinstance(call_args[0][0].update, AgentPlanUpdate)
    ]
    assert plan_updates, "Expected an AgentPlanUpdate from planning tool"
    first_entries = plan_updates[0].entries or []
    assert len(first_entries) == 2
    assert first_entries[0].content == "alpha"
    assert first_entries[1].content == "beta"
