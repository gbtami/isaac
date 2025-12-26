from __future__ import annotations

from types import SimpleNamespace
import importlib
from unittest.mock import AsyncMock

import pytest
from acp import text_block
from acp.schema import AgentPlanUpdate
from pydantic_ai.messages import FunctionToolCallEvent, FunctionToolResultEvent, ToolCallPart  # type: ignore

from typing import Any

from isaac.agent import ACPAgent
from isaac.agent.brain.prompt_runner import PromptEnv
from isaac.agent.brain.plan_schema import PlanStep, PlanSteps
from isaac.agent.brain.prompt_handler import PromptHandler
from isaac.agent.brain.prompt_handler import SessionState
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


@pytest.mark.asyncio
async def test_subagent_planner_plan_updates(tmp_path, monkeypatch):
    conn = AsyncMock()
    conn.session_update = AsyncMock()
    conn.request_permission = AsyncMock(return_value=True)

    plan = PlanSteps(entries=[PlanStep(content="alpha", priority="high"), PlanStep(content="beta")])
    runner = _FakeRunner(plan)

    from isaac.agent.brain import prompt_handler

    def _build(_model_id: str, _register: object, toolsets=None, **kwargs: object):
        _ = toolsets
        return runner

    monkeypatch.setattr(prompt_handler, "create_subagent_for_model", _build)
    agent = ACPAgent(conn)

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

    from isaac.agent.brain import prompt_handler

    def _build(_model_id: str, _register: object, toolsets=None, **kwargs: object):
        _ = toolsets
        return runner

    monkeypatch.setattr(prompt_handler, "create_subagent_for_model", _build)
    agent = ACPAgent(conn)

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
@pytest.mark.parametrize("tool_name", ["planner", "review", "coding"])
async def test_delegate_tool_isolation_default(monkeypatch, tool_name: str):
    captured: dict[str, Any] = {}

    async def fake_stream_with_runner(
        _runner: Any,
        prompt: str,
        *_: object,
        history: list[Any] | None = None,
        **__: object,
    ):
        captured["prompt"] = prompt
        captured["history"] = history
        return "ok", None

    class RunnerStub:
        def run_stream_events(self, *_: Any, **__: Any):
            async def _gen():
                yield "ok"

            return _gen()

    monkeypatch.setattr("isaac.agent.subagents.delegate_tools.stream_with_runner", fake_stream_with_runner)
    monkeypatch.setattr(
        "isaac.agent.subagents.delegate_tools._build_delegate_agent",
        lambda *_args, **_kwargs: RunnerStub(),
    )

    module = importlib.import_module(f"isaac.agent.subagents.{tool_name}")
    tool = getattr(module, tool_name)
    result = await tool(None, task="check isolation")

    assert result["error"] is None
    assert result["content"] == "ok"
    assert captured["history"] is None
    assert captured["prompt"] == "check isolation"


@pytest.mark.asyncio
async def test_subagent_history_compaction(monkeypatch):
    monkeypatch.setattr(PromptHandler, "_MAX_HISTORY_MESSAGES", 3)
    monkeypatch.setattr(PromptHandler, "_PRESERVE_RECENT_MESSAGES", 1)

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

    monkeypatch.setattr("isaac.agent.brain.prompt_handler.stream_with_runner", fake_stream_with_runner)

    env = PromptEnv(
        session_modes={},
        session_last_chunk={},
        send_update=AsyncMock(),
        request_run_permission=AsyncMock(return_value=True),
        set_usage=lambda *_: None,
    )
    handler = PromptHandler(env, register_tools=lambda *_: None)
    state = SessionState(
        runner=object(),
        model_id="m",
        history=[
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
            {"role": "assistant", "content": "d"},
        ],
    )

    await handler._maybe_compact_history(state)

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

    from isaac.agent.brain import prompt_handler

    def _build(_model_id: str, _register: object, toolsets=None, **kwargs: object):
        _ = toolsets
        return runner

    monkeypatch.setattr(prompt_handler, "create_subagent_for_model", _build)
    agent = ACPAgent(conn)

    session = await agent.new_session(cwd=str(tmp_path), mcp_servers=[])
    await agent.prompt(prompt=[text_block("run it")], session_id=session.session_id)

    state = agent._prompt_handler._sessions[session.session_id]  # type: ignore[attr-defined]
    history_text = " ".join(str(msg.get("content") or "") for msg in state.history if isinstance(msg, dict))
    assert "python main.py" in history_text
