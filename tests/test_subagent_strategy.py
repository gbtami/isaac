from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from acp import text_block
from acp.schema import AgentPlanUpdate
from pydantic_ai.messages import FunctionToolCallEvent, FunctionToolResultEvent, ToolCallPart  # type: ignore

from typing import Any

from isaac.agent import ACPAgent
from isaac.agent.brain.strategy_runner import StrategyEnv
from isaac.agent.brain.strategy_plan import PlanStep, PlanSteps
from isaac.agent.brain.subagent_strategy import SubagentPromptStrategy
from isaac.agent.brain.subagent_strategy import SubagentSessionState
from pydantic_ai.run import AgentRunResult, AgentRunResultEvent  # type: ignore


class _FakeRunner:
    def __init__(self, plan: PlanSteps):
        self.prompts: list[str] = []
        self._plan = plan

    def tool(self, *_: object, **__: object):
        def _decorator(func):
            return func

        return _decorator

    async def run_stream_events(self, prompt: str, message_history=None):
        self.prompts.append(prompt)

        async def _gen():
            part = ToolCallPart(tool_name="todo", args={"task": prompt}, tool_call_id="tc1")
            call_event = FunctionToolCallEvent(part=part)
            yield call_event
            result_event = FunctionToolResultEvent(
                result=SimpleNamespace(tool_name="todo", content=self._plan, tool_call_id=part.tool_call_id)
            )
            yield result_event
            yield "done"

        return _gen()


class _PlannerStub:
    async def run(self, prompt: str):  # pragma: no cover - simple stub
        return prompt


@pytest.mark.asyncio
async def test_subagent_todo_plan_updates(tmp_path, monkeypatch):
    conn = AsyncMock()
    conn.session_update = AsyncMock()
    conn.request_permission = AsyncMock(return_value=True)

    plan = PlanSteps(entries=[PlanStep(content="alpha", priority="high"), PlanStep(content="beta")])
    runner = _FakeRunner(plan)

    env = StrategyEnv(
        session_modes={},
        session_last_chunk={},
        send_update=conn.session_update,
        request_run_permission=conn.request_permission,
        set_usage=lambda *_: None,
    )
    from isaac.agent.brain import subagent_strategy

    def _build(_model_id: str, _register: object, toolsets=None):
        _ = toolsets
        return runner

    def _build_planner(_model_id: str):
        return _PlannerStub()

    monkeypatch.setattr(subagent_strategy, "create_subagent_for_model", _build)
    monkeypatch.setattr(subagent_strategy, "create_subagent_planner_for_model", _build_planner)
    strategy = SubagentPromptStrategy(env, register_tools=lambda *_: None)
    agent = ACPAgent(conn, prompt_strategy=strategy)

    session = await agent.new_session(cwd=str(tmp_path), mcp_servers=[])
    await agent.prompt(prompt=[text_block("do work")], session_id=session.session_id)

    updates = []
    for call in conn.session_update.await_args_list:  # type: ignore[attr-defined]
        if "update" in call.kwargs:
            note = call.kwargs["update"]
        elif call.args:
            note = call.args[0]
        else:
            continue
        updates.append(getattr(note, "update", note))
    plan_updates = [u for u in updates if isinstance(u, AgentPlanUpdate)]

    assert plan_updates, "Expected plan updates from todo tool"
    assert plan_updates[0].entries[0].status == "in_progress"
    assert plan_updates[0].entries[0].field_meta.get("id")
    assert plan_updates[-1].entries[0].status == "completed"
    assert runner.prompts  # ensure runner executed


@pytest.mark.asyncio
async def test_subagent_todo_uses_planner_history(monkeypatch):
    # Force small history window for test
    monkeypatch.setattr(SubagentPromptStrategy, "_MAX_HISTORY_MESSAGES", 2)

    captured_histories: list[list[Any]] = []

    async def fake_stream_with_runner(
        _runner: Any,
        _prompt: str,
        *_: object,
        history: list[Any] | None = None,
        on_event=None,
        store_messages=None,
        **__: object,
    ):
        captured_histories.append(list(history or []))
        if store_messages:
            store_messages(["m1"])
        if on_event:
            await on_event(AgentRunResultEvent(result=AgentRunResult(PlanSteps(entries=[PlanStep(content="x")]))))
        return PlanSteps(entries=[PlanStep(content="x")]), None

    monkeypatch.setattr("isaac.agent.brain.subagent_strategy.stream_with_runner", fake_stream_with_runner)

    env = StrategyEnv(
        session_modes={},
        session_last_chunk={},
        send_update=AsyncMock(),
        request_run_permission=AsyncMock(return_value=True),
        set_usage=lambda *_: None,
    )

    class RunnerStub:
        def tool(self, *_: object, **__: object):
            def _decorator(func):
                self.todo_fn = func
                return func

            return _decorator

    strategy = SubagentPromptStrategy(env, register_tools=lambda *_: None)
    state = SubagentSessionState(runner=RunnerStub(), todo_planner=object(), model_id="m")
    strategy._attach_todo_tool(state)

    await state.runner.todo_fn(object(), task="do it")  # type: ignore[attr-defined]
    await state.runner.todo_fn(object(), task="do it")  # second call should carry planner history

    assert captured_histories[0] == []
    assert captured_histories[1] == ["m1"]


@pytest.mark.asyncio
async def test_subagent_history_compaction(monkeypatch):
    monkeypatch.setattr(SubagentPromptStrategy, "_MAX_HISTORY_MESSAGES", 3)
    monkeypatch.setattr(SubagentPromptStrategy, "_PRESERVE_RECENT_MESSAGES", 1)

    compaction_calls: list[list[Any]] = []

    async def fake_stream_with_runner(
        _runner: Any,
        _prompt: str,
        *_: object,
        history: list[Any] | None = None,
        **__: object,
    ):
        compaction_calls.append(list(history or []))
        return "summary here", None

    monkeypatch.setattr("isaac.agent.brain.subagent_strategy.stream_with_runner", fake_stream_with_runner)

    env = StrategyEnv(
        session_modes={},
        session_last_chunk={},
        send_update=AsyncMock(),
        request_run_permission=AsyncMock(return_value=True),
        set_usage=lambda *_: None,
    )
    strategy = SubagentPromptStrategy(env, register_tools=lambda *_: None)
    state = SubagentSessionState(
        runner=object(),
        model_id="m",
        history=[
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
            {"role": "assistant", "content": "d"},
        ],
    )

    await strategy._maybe_compact_history(state)

    assert compaction_calls, "Compaction should have been invoked"
    assert len(state.history) == 2  # summary + preserved message
    assert state.history[0]["role"] == "system"
