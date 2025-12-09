from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from acp import text_block
from acp.schema import AgentMessageChunk, AgentPlanUpdate

from tests.utils import make_function_agent


@pytest.mark.asyncio
async def test_plan_only_strategy_emits_plan_update_without_execution():
    conn = AsyncMock()
    planning_runner = _PlanningRunner("Plan:\n- step one\n- step two")
    executor = _StreamingExecutor()
    agent = make_function_agent(conn)
    agent._planning_runner = planning_runner  # type: ignore[attr-defined]
    agent._ai_runner = executor  # type: ignore[attr-defined]

    session_id = "plan-only"
    agent._session_prompt_strategies[session_id] = "plan_only"
    response = await agent.prompt(prompt=[text_block("plan only")], session_id=session_id)

    updates = [call.kwargs["update"] for call in conn.session_update.await_args_list]  # type: ignore[attr-defined]
    assert any(isinstance(u, AgentPlanUpdate) for u in updates)
    assert executor.prompts == []
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
class _DelegateRunner:
    def __init__(self):
        self.prompts: list[str] = []
        self.tools: dict[str, object] = {}

    def tool(self, name: str):
        def _decorator(fn):
            self.tools[name] = fn
            return fn

        return _decorator

    async def run_stream_events(
        self, prompt: str, message_history: list[dict[str, str]] | None = None, **_: object
    ):
        from pydantic_ai.messages import (
            FunctionToolCallEvent,
            FunctionToolResultEvent,
            ToolCallPart,
            ToolReturnPart,
        )

        self.prompts.append(prompt)
        call = FunctionToolCallEvent(
            ToolCallPart(tool_name="delegate_plan", args={"task": prompt}, tool_call_id="call1")
        )
        result = FunctionToolResultEvent(
            ToolReturnPart(
                tool_name="delegate_plan",
                content={"plan": ["step1", "step2"], "tool": "delegate_plan"},
                tool_call_id="call1",
            )
        )

        async def _gen():
            yield call
            yield result
            yield "delegate complete"

        return _gen()


class _FailingPlanner:
    def __init__(self):
        self.called = False

    async def run_stream_events(self, prompt: str, **_: object):
        self.called = True
        raise AssertionError("planner should not be used in single strategy")


class _SingleRunner:
    def __init__(self):
        self.prompts: list[str] = []

    async def run_stream_events(
        self, prompt: str, message_history: list[dict[str, str]] | None = None, **_: object
    ):
        self.prompts.append(prompt)

        async def _gen():
            yield "plan then execute"

        return _gen()


@pytest.mark.asyncio
async def test_delegation_strategy_emits_plan_update_and_execution():
    conn = AsyncMock()
    delegate_runner = _DelegateRunner()
    agent = make_function_agent(conn)
    agent._ai_runner = delegate_runner  # type: ignore[attr-defined]
    session_id = "delegation-session"
    agent._session_prompt_strategies[session_id] = "delegation"

    await agent.prompt(prompt=[text_block("do delegated work")], session_id=session_id)

    updates = [call.kwargs["update"] for call in conn.session_update.await_args_list]  # type: ignore[attr-defined]
    assert any(isinstance(u, AgentPlanUpdate) for u in updates)
    assert any(
        isinstance(u, AgentMessageChunk)
        and getattr(getattr(u, "content", None), "text", "") == "delegate complete"
        for u in updates
    )
    assert len(delegate_runner.prompts) == 1
    assert "delegate_plan" in delegate_runner.prompts[0]


@pytest.mark.asyncio
async def test_single_strategy_uses_single_runner_only():
    conn = AsyncMock()
    single_runner = _SingleRunner()
    failing_planner = _FailingPlanner()
    agent = make_function_agent(conn)
    agent._ai_runner = single_runner  # type: ignore[attr-defined]
    agent._planning_runner = failing_planner  # type: ignore[attr-defined]
    session_id = "single-session"
    agent._session_prompt_strategies[session_id] = "single"

    await agent.prompt(prompt=[text_block("do single agent work")], session_id=session_id)

    assert len(single_runner.prompts) == 1
    assert "share a short plan" in single_runner.prompts[0]
    assert not failing_planner.called
    updates = [call.kwargs["update"] for call in conn.session_update.await_args_list]  # type: ignore[attr-defined]
    assert any(
        isinstance(u, AgentMessageChunk)
        and getattr(getattr(u, "content", None), "text", "") == "plan then execute"
        for u in updates
    )
