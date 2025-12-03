from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from acp import PromptRequest, text_block
from acp.schema import AgentMessageChunk, AgentPlanUpdate

from tests.utils import make_function_agent


@pytest.mark.asyncio
async def test_plan_updates_for_plan_prefix():
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


class _PlanningRunner:
    def __init__(self, content: str):
        self.prompts: list[str] = []
        self.content = content

    async def run(self, prompt: str, messages=None):
        self.prompts.append(prompt)
        return SimpleNamespace(output=self.content)


class _StreamingExecutor:
    def __init__(self):
        self.prompts: list[str] = []

    async def run_stream_events(self, prompt: str, messages=None):
        self.prompts.append(prompt)

        async def _gen():
            yield "executed"

        return _gen()


@pytest.mark.asyncio
async def test_programmatic_plan_then_execute():
    conn = AsyncMock()
    planning_runner = _PlanningRunner("Plan:\n- alpha\n- beta")
    executor = _StreamingExecutor()
    agent = make_function_agent(conn)
    agent._planning_runner = planning_runner  # type: ignore[attr-defined]
    agent._ai_runner = executor  # type: ignore[attr-defined]

    session_id = "handoff"
    await agent.prompt(PromptRequest(sessionId=session_id, prompt=[text_block("do work")]))

    updates = [call.args[0].update for call in conn.sessionUpdate.await_args_list]  # type: ignore[attr-defined]
    assert any(isinstance(u, AgentPlanUpdate) for u in updates)
    assert any(
        isinstance(u, AgentMessageChunk)
        and getattr(getattr(u, "content", None), "text", "") == "executed"
        for u in updates
    )
    assert len(planning_runner.prompts) == 1
    assert len(executor.prompts) == 1
    assert "Plan:" in executor.prompts[0]


@pytest.mark.asyncio
async def test_plan_only_prompt_skips_execution():
    conn = AsyncMock()
    planning_runner = _PlanningRunner("Plan:\n- alpha\n- beta")
    executor = _StreamingExecutor()
    agent = make_function_agent(conn)
    agent._planning_runner = planning_runner  # type: ignore[attr-defined]
    agent._ai_runner = executor  # type: ignore[attr-defined]

    session_id = "plan-only"
    await agent.prompt(
        PromptRequest(sessionId=session_id, prompt=[text_block("Just a plan only, no execution.")])
    )

    assert len(planning_runner.prompts) == 1
    assert executor.prompts == []
