from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
from acp import text_block
from acp.schema import AgentMessageChunk, AgentPlanUpdate
from pydantic_ai.run import AgentRunResult, AgentRunResultEvent  # type: ignore

from isaac.agent.brain.handoff import HandoffEnv, HandoffPromptRunner, PlanStep, PlanSteps
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
    env = HandoffEnv(
        session_modes={},
        session_last_chunk={},
        send_update=send_update,
        request_run_permission=AsyncMock(return_value=True),
        set_usage=lambda *_: None,
    )
    runner = HandoffPromptRunner(env)

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

    assert plan_update is not None
    assert plan_text == "- a"
    entries = getattr(plan_update, "entries", []) or []
    assert [e.content for e in entries] == ["a"]
    assert [e.priority for e in entries] == ["high"]
    assert all(getattr(e, "status", "") == "pending" for e in entries)


@pytest.mark.asyncio
async def test_structured_plan_reaches_executor_prompt():
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

    agent._planning_runner = _StructuredPlanner()  # type: ignore[attr-defined]
    agent._ai_runner = _RecordingExecutor()  # type: ignore[attr-defined]
    session_id = "structured-handoff"

    await agent.prompt(prompt=[text_block("ship the feature")], session_id=session_id)

    updates = [call.kwargs["update"] for call in conn.session_update.await_args_list]  # type: ignore[attr-defined]
    plan_updates = [u for u in updates if isinstance(u, AgentPlanUpdate)]
    assert len(plan_updates) == 2
    assert plan_updates[0].entries[0].status == "in_progress"
    assert plan_updates[0].entries[0].priority == "high"
    assert plan_updates[-1].entries and all(e.status == "completed" for e in plan_updates[-1].entries)

    executor_prompts = agent._ai_runner.prompts  # type: ignore[attr-defined]
    assert executor_prompts
    prompt_text = executor_prompts[0]
    assert "Plan:" in prompt_text
    assert "- alpha" in prompt_text
    assert "- beta" in prompt_text
    assert "Execute this plan now" in prompt_text


def test_plan_parser_handles_steps_list_line():
    from isaac.agent.brain.planner import parse_plan_from_text

    update = parse_plan_from_text("steps=['first','second','third']")
    assert update
    contents = [getattr(e, "content", "") for e in update.entries]
    assert contents == ["first", "second", "third"]
