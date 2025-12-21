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
            part = ToolCallPart(tool_name="planner", args={"task": prompt}, tool_call_id="tc1")
            call_event = FunctionToolCallEvent(part=part)
            yield call_event
            result_event = FunctionToolResultEvent(
                result=SimpleNamespace(tool_name="planner", content=self._plan, tool_call_id=part.tool_call_id)
            )
            yield result_event
            yield "done"

        return _gen()

    def set_plan(self, plan: PlanSteps) -> None:
        self._plan = plan


class _PlannerStub:
    async def run(self, prompt: str):  # pragma: no cover - simple stub
        return prompt


@pytest.mark.asyncio
async def test_subagent_planner_plan_updates(tmp_path, monkeypatch):
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

    def _build(_model_id: str, _register: object, toolsets=None, **kwargs: object):
        _ = toolsets
        return runner

    def _build_planner(_model_id: str, **kwargs: object):
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

    assert plan_updates, "Expected plan updates from planner tool"
    assert plan_updates[0].entries[0].status == "in_progress"
    assert plan_updates[0].entries[0].field_meta.get("id")
    assert plan_updates[-1].entries[0].status == "completed"
    assert runner.prompts  # ensure runner executed


@pytest.mark.asyncio
async def test_subagent_plan_refreshes_each_prompt(tmp_path, monkeypatch):
    conn = AsyncMock()
    conn.session_update = AsyncMock()
    conn.request_permission = AsyncMock(return_value=True)

    plan1 = PlanSteps(entries=[PlanStep(content="only")])
    plan2 = PlanSteps(entries=[PlanStep(content="first"), PlanStep(content="second")])
    runner = _FakeRunner(plan1)

    class _PlannerRunner:
        def __init__(self, plan: PlanSteps):
            self.plan = plan

        async def run_stream_events(self, prompt: str, message_history=None):
            _ = prompt, message_history

            async def _gen():
                yield AgentRunResultEvent(result=AgentRunResult(self.plan))

            return _gen()

        def set_plan(self, plan: PlanSteps) -> None:
            self.plan = plan

    planner = _PlannerRunner(plan1)

    env = StrategyEnv(
        session_modes={},
        session_last_chunk={},
        send_update=conn.session_update,
        request_run_permission=conn.request_permission,
        set_usage=lambda *_: None,
    )
    from isaac.agent.brain import subagent_strategy

    def _build(_model_id: str, _register: object, toolsets=None, **kwargs: object):
        _ = toolsets
        return runner

    def _build_planner(_model_id: str, **kwargs: object):
        return planner

    monkeypatch.setattr(subagent_strategy, "create_subagent_for_model", _build)
    monkeypatch.setattr(subagent_strategy, "create_subagent_planner_for_model", _build_planner)
    strategy = SubagentPromptStrategy(env, register_tools=lambda *_: None)
    agent = ACPAgent(conn, prompt_strategy=strategy)

    session = await agent.new_session(cwd=str(tmp_path), mcp_servers=[])
    await agent.prompt(prompt=[text_block("first")], session_id=session.session_id)
    updates_first: list[Any] = []
    for call in conn.session_update.await_args_list:  # type: ignore[attr-defined]
        if "update" in call.kwargs:
            note = call.kwargs["update"]
        elif call.args:
            note = call.args[0]
        else:
            continue
        updates_first.append(getattr(note, "update", note))
    conn.session_update.reset_mock()

    runner.set_plan(plan2)
    planner.set_plan(plan2)
    await agent.prompt(prompt=[text_block("second")], session_id=session.session_id)
    updates_second: list[Any] = []
    for call in conn.session_update.await_args_list:  # type: ignore[attr-defined]
        if "update" in call.kwargs:
            note = call.kwargs["update"]
        elif call.args:
            note = call.args[0]
        else:
            continue
        updates_second.append(getattr(note, "update", note))
    plan_updates_second = [u for u in updates_second if isinstance(u, AgentPlanUpdate)]
    assert plan_updates_second, "Expected plan update on second prompt"
    assert len(plan_updates_second[-1].entries) == 2, "Expected refreshed multi-step plan"


@pytest.mark.asyncio
async def test_subagent_planner_resets_planner_history(monkeypatch):
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
                self.planner_fn = func
                return func

            return _decorator

    strategy = SubagentPromptStrategy(env, register_tools=lambda *_: None)
    state = SubagentSessionState(runner=RunnerStub(), planner=object(), model_id="m")
    strategy._attach_planner_tool(state)

    await state.runner.planner_fn(object(), task="do it")  # type: ignore[attr-defined]
    await state.runner.planner_fn(object(), task="do it")  # second call should not carry prior planner content

    assert captured_histories[0] == []
    assert captured_histories[1] == []  # planner history is reset between calls


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


@pytest.mark.asyncio
async def test_subagent_records_tool_history(tmp_path, monkeypatch):
    conn = AsyncMock()
    conn.session_update = AsyncMock()
    conn.request_permission = AsyncMock(return_value=True)

    class CommandRunner:
        def tool(self, *_: object, **__: object):
            def _decorator(func):
                return func

            return _decorator

        async def run_stream_events(self, prompt: str, message_history=None):
            _ = prompt, message_history
            part = ToolCallPart(
                tool_name="run_command",
                args={"command": "python main.py", "cwd": str(tmp_path)},
                tool_call_id="tc_hist",
            )

            async def _gen():
                yield FunctionToolCallEvent(part=part)
                result_part = SimpleNamespace(
                    tool_name="run_command",
                    content={"command": "python main.py", "cwd": str(tmp_path), "stdout": "ok"},
                    tool_call_id=part.tool_call_id,
                )
                yield FunctionToolResultEvent(result=result_part)
                yield AgentRunResultEvent(result=AgentRunResult("done"))

            return _gen()

    runner = CommandRunner()

    env = StrategyEnv(
        session_modes={},
        session_last_chunk={},
        send_update=conn.session_update,
        request_run_permission=conn.request_permission,
        set_usage=lambda *_: None,
    )
    from isaac.agent.brain import subagent_strategy

    def _build(_model_id: str, _register: object, toolsets=None, **kwargs: object):
        _ = toolsets
        return runner

    monkeypatch.setattr(subagent_strategy, "create_subagent_for_model", _build)
    monkeypatch.setattr(subagent_strategy, "create_subagent_planner_for_model", lambda *_args, **_kwargs: object())

    strategy = SubagentPromptStrategy(env, register_tools=lambda *_: None)
    agent = ACPAgent(conn, prompt_strategy=strategy)

    session = await agent.new_session(cwd=str(tmp_path), mcp_servers=[])
    await agent.prompt(prompt=[text_block("run it")], session_id=session.session_id)

    state = strategy._sessions[session.session_id]  # type: ignore[attr-defined]
    history_text = " ".join(str(msg.get("content") or "") for msg in state.history if isinstance(msg, dict))
    assert "python main.py" in history_text
