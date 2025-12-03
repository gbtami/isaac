from __future__ import annotations

import pytest
from pydantic_ai import Agent as PydanticAgent  # type: ignore
from pydantic_ai.models.test import TestModel  # type: ignore

from isaac.agent.brain.planning_delegate import make_planning_tool


@pytest.mark.asyncio
async def test_planning_tool_accepts_plan_keyword():
    planning_agent = PydanticAgent(TestModel(call_tools=[]))
    tool = make_planning_tool(planning_agent)
    result = await tool(plan="ship feature", context="short")
    assert result.get("plan_steps")
