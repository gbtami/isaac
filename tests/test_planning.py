from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
from acp import text_block
from acp.schema import AgentMessageChunk, AgentPlanUpdate
from pydantic_ai.messages import FunctionToolCallEvent, ToolCallPart  # type: ignore
from pydantic_ai.run import AgentRunResult, AgentRunResultEvent  # type: ignore

from isaac.agent.brain.strategy_runner import StrategyEnv
from isaac.agent.brain.handoff_runner import HandoffRunner
from isaac.agent.brain.strategy_plan import PlanStep, PlanSteps
from tests.utils import make_function_agent


class _PlanningRunner:
    def __init__(self, content: str):
        self.prompts: list[str] = []
        self.content = content

    async def run_stream_events(self, prompt: str, message_history: list[dict[str, str]] | None = None, **_: object):
        self.prompts.append(prompt)

        async def _gen():
            yield self.content

        return _gen()


class _StreamingExecutor:
    def __init__(self):
        self.prompts: list[str] = []

    async def run_stream_events(self, prompt: str, message_history: list[dict[str, str]] | None = None, **_: object):
        self.prompts.append(prompt)

        async def _gen():
            yield "executed"

        return _gen()


@pytest.mark.asyncio
async def test_programmatic_plan_then_execute(monkeypatch):
    conn = AsyncMock()
    planning_runner = _PlanningRunner("Plan:\n- alpha\n- beta")
    executor = _StreamingExecutor()
    agent = make_function_agent(conn)
    from isaac.agent.brain import handoff_strategy

    def _build(_model_id: str, _register: object, toolsets=None) -> tuple[object, object]:
        _ = toolsets
        return executor, planning_runner

    monkeypatch.setattr(handoff_strategy, "create_agents_for_model", _build)
    session_id = "handoff"
    await agent.prompt(prompt=[text_block("do work")], session_id=session_id)

    updates = [call.kwargs["update"] for call in conn.session_update.await_args_list]  # type: ignore[attr-defined]
    plan_updates = [u for u in updates if isinstance(u, AgentPlanUpdate)]
    assert plan_updates
    assert any(
        isinstance(u, AgentMessageChunk) and getattr(getattr(u, "content", None), "text", "") == "executed"
        for u in updates
    )
    assert any(e.status == "in_progress" for e in plan_updates[0].entries)
    assert plan_updates[-1].entries and all(e.status == "completed" for e in plan_updates[-1].entries)
    assert len(planning_runner.prompts) == 1
    assert len(executor.prompts) == 1
    assert "Plan:" in executor.prompts[0]


@pytest.mark.asyncio
async def test_structured_plan_entries_generate_plan_updates():
    send_update = AsyncMock()
    env = StrategyEnv(
        session_modes={},
        session_last_chunk={},
        send_update=send_update,
        request_run_permission=AsyncMock(return_value=True),
        set_usage=lambda *_: None,
    )
    runner = HandoffRunner(env)

    class _StructuredPlanner:
        async def run_stream_events(
            self, prompt: str, message_history: list[dict[str, str]] | None = None, **_: object
        ):
            _ = prompt, message_history

            class _Result:
                def __init__(self, output: PlanSteps):
                    self.output = output
                    self.data = output

            event = AgentRunResultEvent(result=_Result(PlanSteps(entries=[PlanStep(content="a", priority="high")])))

            async def _gen():
                yield event

            return _gen()

    plan_update, plan_text, _ = await runner._run_planning_phase(
        "s1",
        "do work",
        history=[],
        cancel_event=asyncio.Event(),
        planner=_StructuredPlanner(),
        store_model_messages=lambda *_: None,
    )

    assert plan_update is None, "single-step plans should not emit plan updates"
    assert plan_text == "- a"


@pytest.mark.asyncio
async def test_structured_plan_reaches_executor_prompt(monkeypatch):
    conn = AsyncMock()
    agent = make_function_agent(conn)

    class _StructuredPlanner:
        def __init__(self):
            self.prompts: list[str] = []

        async def run_stream_events(
            self, prompt: str, message_history: list[dict[str, str]] | None = None, **_: object
        ):
            self.prompts.append(prompt)
            event = AgentRunResultEvent(
                result=AgentRunResult(
                    PlanSteps(entries=[PlanStep(content="alpha", priority="high"), PlanStep(content="beta")])
                )
            )

            async def _gen():
                yield event

            return _gen()

    class _RecordingExecutor:
        def __init__(self):
            self.prompts: list[str] = []

        async def run_stream_events(
            self, prompt: str, message_history: list[dict[str, str]] | None = None, **_: object
        ):
            self.prompts.append(prompt)

            async def _gen():
                yield "done"

            return _gen()

    executor = _RecordingExecutor()
    planner = _StructuredPlanner()
    from isaac.agent.brain import handoff_strategy

    def _build(_model_id: str, _register: object, toolsets=None) -> tuple[object, object]:
        _ = toolsets
        return executor, planner

    monkeypatch.setattr(handoff_strategy, "create_agents_for_model", _build)
    session_id = "structured-handoff"

    await agent.prompt(prompt=[text_block("ship the feature")], session_id=session_id)

    updates = [call.kwargs["update"] for call in conn.session_update.await_args_list]  # type: ignore[attr-defined]
    plan_updates = [u for u in updates if isinstance(u, AgentPlanUpdate)]
    assert len(plan_updates) == 2
    assert plan_updates[0].entries[0].status == "in_progress"
    assert plan_updates[0].entries[0].priority == "high"
    assert plan_updates[-1].entries and all(e.status == "completed" for e in plan_updates[-1].entries)

    assert executor.prompts
    prompt_text = executor.prompts[0]
    assert "Plan:" in prompt_text
    assert "- alpha" in prompt_text
    assert "- beta" in prompt_text
    assert "Execute this plan now" in prompt_text


@pytest.mark.asyncio
async def test_single_step_plan_skips_plan_updates(monkeypatch):
    conn = AsyncMock()
    conn.session_update = AsyncMock()
    agent = make_function_agent(conn)

    class _SingleStepPlanner:
        def __init__(self):
            self.prompts: list[str] = []

        async def run_stream_events(
            self, prompt: str, message_history: list[dict[str, str]] | None = None, **_: object
        ):
            self.prompts.append(prompt)
            event = AgentRunResultEvent(result=AgentRunResult(PlanSteps(entries=[PlanStep(content="solo")])))

            async def _gen():
                yield event

            return _gen()

    class _RecordingExecutor:
        def __init__(self):
            self.prompts: list[str] = []

        async def run_stream_events(
            self, prompt: str, message_history: list[dict[str, str]] | None = None, **_: object
        ):
            self.prompts.append(prompt)

            async def _gen():
                yield "done"

            return _gen()

    executor = _RecordingExecutor()
    planner = _SingleStepPlanner()
    from isaac.agent.brain import handoff_strategy

    def _build(_model_id: str, _register: object, toolsets=None) -> tuple[object, object]:
        _ = toolsets
        return executor, planner

    monkeypatch.setattr(handoff_strategy, "create_agents_for_model", _build)
    session = await agent.new_session(cwd="/", mcp_servers=[])

    await agent.prompt(prompt=[text_block("do one thing")], session_id=session.session_id)

    updates = [call.kwargs["update"] for call in conn.session_update.await_args_list]  # type: ignore[attr-defined]
    plan_updates = [u for u in updates if isinstance(u, AgentPlanUpdate)]
    assert plan_updates == [], "Single-step plans should not emit plan updates"
    assert executor.prompts
    assert "- solo" in executor.prompts[0]


def test_plan_parser_handles_steps_list_line():
    from isaac.agent.brain.planner import parse_plan_from_text

    update = parse_plan_from_text("steps=['first','second','third']")
    assert update
    contents = [getattr(e, "content", "") for e in update.entries]
    assert contents == ["first", "second", "third"]


@pytest.mark.asyncio
async def test_run_command_permission_includes_string_args():
    send_update = AsyncMock()
    request_perm = AsyncMock(return_value=True)
    env = StrategyEnv(
        session_modes={"s": "ask"},
        session_last_chunk={},
        send_update=send_update,
        request_run_permission=request_perm,
        set_usage=lambda *_: None,
    )
    runner = HandoffRunner(env)
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
