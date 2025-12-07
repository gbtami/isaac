from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from acp import text_block
from acp.schema import AgentMessageChunk, AgentPlanUpdate

from tests.utils import make_function_agent


@pytest.mark.asyncio
async def test_plan_updates_for_plan_prefix():
    conn = AsyncMock()
    agent = make_function_agent(conn)

    session_id = "plan-session"
    response = await agent.prompt(
        prompt=[text_block("plan:step one;step two")], session_id=session_id
    )

    conn.session_update.assert_called_once()
    update = conn.session_update.call_args.kwargs["update"]
    assert hasattr(update, "entries")
    assert len(update.entries) == 2
    assert response.stop_reason == "end_turn"


class _PlanningRunner:
    def __init__(self, content: str):
        self.prompts: list[str] = []
        self.content = content

    async def run_stream_events(
        self, prompt: str, message_history: list[dict[str, str]] | None = None, **_: object
    ):
        self.prompts.append(prompt)

        async def _gen():
            yield self.content

        return _gen()


class _StreamingExecutor:
    def __init__(self):
        self.prompts: list[str] = []

    async def run_stream_events(
        self, prompt: str, message_history: list[dict[str, str]] | None = None, **_: object
    ):
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
    await agent.prompt(prompt=[text_block("do work")], session_id=session_id)

    updates = [call.kwargs["update"] for call in conn.session_update.await_args_list]  # type: ignore[attr-defined]
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
        prompt=[text_block("Just a plan only, no execution.")], session_id=session_id
    )

    assert len(planning_runner.prompts) == 1
    assert executor.prompts == []
