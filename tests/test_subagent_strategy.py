from __future__ import annotations

from types import SimpleNamespace
import contextlib
import importlib
import json
import re
from unittest.mock import AsyncMock

import pytest
from acp import text_block
from acp.schema import AgentPlanUpdate, AgentThoughtChunk, ToolCallProgress
from pydantic_ai.messages import (  # type: ignore
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ToolCallPart,
)
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolReturnPart  # type: ignore
from pydantic_ai import Agent as PydanticAgent  # type: ignore
from pydantic_ai import DeferredToolRequests  # type: ignore
from pydantic_ai.models.test import TestModel  # type: ignore

from typing import Any

from isaac.agent import ACPAgent
from isaac.agent.brain.prompt_runner import PromptEnv
from isaac.agent.brain.plan_schema import PlanStep, PlanSteps
from isaac.agent.brain import compaction as compaction_utils
from isaac.agent.brain.session_state import SessionState
from pydantic_ai.run import AgentRunResult, AgentRunResultEvent  # type: ignore
from isaac.agent.tools import register_tools
from isaac.agent.subagents.delegate_tools import (
    DelegateToolContext,
    reset_delegate_tool_context,
    set_delegate_tool_context,
)


COMPACT_USER_MAX_TOKENS = 20_000


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

    from isaac.agent.brain import session_ops

    def _build(_model_id: str, _register: object, toolsets=None, **kwargs: object):
        _ = toolsets
        return runner

    monkeypatch.setattr(session_ops, "create_subagent_for_model", _build)
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

    from isaac.agent.brain import session_ops

    def _build(_model_id: str, _register: object, toolsets=None, **kwargs: object):
        _ = toolsets
        return runner

    monkeypatch.setattr(session_ops, "create_subagent_for_model", _build)
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
    assert "check isolation" in captured["prompt"]


@pytest.mark.asyncio
async def test_subagent_history_compaction(monkeypatch):
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

    monkeypatch.setattr("isaac.agent.brain.compaction.stream_with_runner", fake_stream_with_runner)

    noop = AsyncMock()
    env = PromptEnv(
        session_modes={},
        session_last_chunk={},
        send_message_chunk=noop,
        send_thought_chunk=noop,
        send_tool_start=noop,
        send_tool_finish=noop,
        send_plan_update=noop,
        send_notification=noop,
        send_protocol_update=noop,
        request_run_permission=AsyncMock(return_value=True),
        set_usage=lambda *_: None,
    )
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

    await compaction_utils.maybe_compact_history(
        env=env,
        state=state,
        session_id=None,
        model_id=state.model_id,
        max_history_messages=3,
        auto_compact_ratio=0.9,
        compact_user_message_max_tokens=COMPACT_USER_MAX_TOKENS,
    )

    assert compaction_calls, "Compaction should have been invoked"
    assert len(state.history) == 3  # user messages + summary
    assert state.history[-1]["role"] == "user"
    assert "summary" in state.history[-1]["content"].lower()


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

    from isaac.agent.brain import session_ops

    def _build(_model_id: str, _register: object, toolsets=None, **kwargs: object):
        _ = toolsets
        return runner

    monkeypatch.setattr(session_ops, "create_subagent_for_model", _build)
    agent = ACPAgent(conn)

    session = await agent.new_session(cwd=str(tmp_path), mcp_servers=[])
    await agent.prompt(prompt=[text_block("run it")], session_id=session.session_id)

    state = agent._prompt_handler._sessions[session.session_id]  # type: ignore[attr-defined]
    history_text = " ".join(str(msg.get("content") or "") for msg in state.history if isinstance(msg, dict))
    assert "python main.py" in history_text


@pytest.mark.asyncio
async def test_delegate_tool_carryover_summary(monkeypatch):
    prompts: list[str] = []

    def _json_summary(text: str) -> str:
        return f'{{"summary": "{text}", "files": [], "tests": [], "risks": [], "followups": []}}'

    long_summary = "Summary: " + ("detail " * 20)

    async def fake_stream_with_runner(
        _runner: Any,
        prompt: str,
        *_: object,
        history: list[Any] | None = None,
        **__: object,
    ):
        prompts.append(prompt)
        assert history is None
        return _json_summary(long_summary), None

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

    from isaac.agent.subagents.coding import coding

    session_id = "carryover-demo"
    first = await coding(None, task="first task", session_id=session_id, carryover=False)
    second = await coding(None, task="second task", session_id=session_id, carryover=True)

    assert first["error"] is None
    assert second["error"] is None
    assert first["delegate_session_id"] == session_id
    assert second["delegate_session_id"] == session_id
    first_content = first["content"]
    if isinstance(first_content, str):
        first_content = json.loads(first_content)
    second_content = second["content"]
    if isinstance(second_content, str):
        second_content = json.loads(second_content)
    assert isinstance(first_content, dict)
    assert isinstance(second_content, dict)

    assert len(prompts) == 2
    assert prompts[0] == "first task"
    assert "Previous delegate summary" in prompts[1]
    assert "Task: second task" in prompts[1]


@pytest.mark.asyncio
async def test_delegate_tool_carryover_acp_integration(monkeypatch, tmp_path):
    """Exercise delegate carryover through ACP tool calls without stubbing the stream runner."""

    class FixedArgsModel(TestModel):  # type: ignore[misc]
        """Return predetermined tool args for the requested function tool."""

        def __init__(self, fixed_args: dict[str, dict[str, object]], **kwargs: object) -> None:
            super().__init__(**kwargs)
            self._fixed_args = fixed_args

        def gen_tool_args(self, tool_def: object) -> dict:
            name = getattr(tool_def, "name", "")
            if name in self._fixed_args:
                return self._fixed_args[name]
            return super().gen_tool_args(tool_def)

        def _request(self, messages, model_settings, model_request_parameters):  # type: ignore[override]
            tool_calls = self._get_tool_calls(model_request_parameters)
            has_tool_returns = any(
                isinstance(message, ModelRequest) and any(isinstance(part, ToolReturnPart) for part in message.parts)
                for message in messages
            )
            if tool_calls and not has_tool_returns:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            name,
                            self.gen_tool_args(args),
                            tool_call_id=f"pyd_ai_tool_call_id__{name}",
                        )
                        for name, args in tool_calls
                    ],
                    model_name=self._model_name,
                )
            return super()._request(messages, model_settings, model_request_parameters)

    class PromptAwareModel(TestModel):  # type: ignore[misc]
        """Return different JSON summaries depending on carryover prompt content."""

        def _request(self, messages, model_settings, model_request_parameters):  # type: ignore[override]
            prompt_text = []
            for message in messages:
                if not isinstance(message, ModelRequest):
                    continue
                for part in message.parts:
                    content = getattr(part, "content", None)
                    if isinstance(content, str):
                        prompt_text.append(content)
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, str):
                                prompt_text.append(item)
            text = "\n".join(prompt_text)
            summary = "carryover seen" if "Previous delegate summary" in text else "no carryover"
            payload = {
                "summary": summary,
                "files": [],
                "tests": [],
                "risks": [],
                "followups": [],
            }
            output_tools = getattr(model_request_parameters, "output_tools", None)
            if output_tools:
                output_tool = output_tools[0]
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            output_tool.name,
                            payload,
                            tool_call_id="delegate-output",
                        )
                    ],
                    model_name=self._model_name,
                )
            return ModelResponse(parts=[TextPart(json.dumps(payload))], model_name=self._model_name)

    from isaac.agent.subagents import delegate_tools as delegate_mod

    def _build_delegate_agent(spec, *_, **__):
        model = PromptAwareModel(call_tools=[])
        agent = PydanticAgent(
            model,
            toolsets=(),
            system_prompt=spec.system_prompt or "test",
            instructions=spec.instructions,
            output_type=[spec.output_type or str, DeferredToolRequests],
        )
        register_tools(agent, tool_names=delegate_mod._expand_tool_names(spec))
        return agent

    monkeypatch.setattr(delegate_mod, "_build_delegate_agent", _build_delegate_agent)

    conn = AsyncMock()
    conn.session_update = AsyncMock()
    agent = ACPAgent(conn)
    session = await agent.new_session(cwd=str(tmp_path), mcp_servers=[])

    runner_first = PydanticAgent(
        FixedArgsModel(
            fixed_args={
                "coding": {
                    "task": "first task",
                    "session_id": "carryover-acp",
                    "carryover": False,
                }
            },
            call_tools=["coding"],
            custom_output_text="done",
        ),
        output_type=[str, DeferredToolRequests],
    )
    register_tools(runner_first)
    agent._prompt_handler.set_session_runner(session.session_id, runner_first)  # type: ignore[attr-defined]

    await agent.prompt(prompt=[text_block("first")], session_id=session.session_id)

    def _extract_summary() -> str | None:
        updates = [
            call.kwargs["update"]
            for call in conn.session_update.call_args_list
            if isinstance(call.kwargs.get("update"), ToolCallProgress)
        ]
        for update in updates:
            raw_output = getattr(update, "raw_output", {}) or {}
            if raw_output.get("delegate_tool") == "coding":
                content = raw_output.get("content")
                if isinstance(content, dict):
                    return content.get("summary")
                if hasattr(content, "summary"):
                    return getattr(content, "summary")
                if isinstance(content, str):
                    match = re.search(r"summary=['\"]([^'\"]+)['\"]", content)
                    if match:
                        return match.group(1)
                    with contextlib.suppress(Exception):
                        parsed = json.loads(content)
                        if isinstance(parsed, dict):
                            return parsed.get("summary")
        return None

    assert _extract_summary() == "no carryover"

    conn.session_update.reset_mock()
    runner_second = PydanticAgent(
        FixedArgsModel(
            fixed_args={
                "coding": {
                    "task": "second task",
                    "session_id": "carryover-acp",
                    "carryover": True,
                }
            },
            call_tools=["coding"],
            custom_output_text="done",
        ),
        output_type=[str, DeferredToolRequests],
    )
    register_tools(runner_second)
    agent._prompt_handler.set_session_runner(session.session_id, runner_second)  # type: ignore[attr-defined]

    await agent.prompt(prompt=[text_block("second")], session_id=session.session_id)

    assert _extract_summary() == "carryover seen"


@pytest.mark.asyncio
async def test_delegate_tool_emits_thought_without_text_progress(monkeypatch):
    updates: list[Any] = []

    async def _capture_update(note: Any) -> None:
        updates.append(getattr(note, "update", note))

    send_update = AsyncMock(side_effect=_capture_update)
    ctx = set_delegate_tool_context(
        DelegateToolContext(
            session_id="s-delegate",
            request_run_permission=AsyncMock(return_value=True),
            send_update=send_update,
            mode_getter=lambda: "ask",
        )
    )
    try:

        async def fake_stream_with_runner(
            _runner: Any,
            _prompt: str,
            on_text: Any,
            on_thought: Any,
            _cancel_event: Any,
            **__: object,
        ):
            assert on_thought is not None
            await on_thought("delegate thinking one ")
            await on_thought("delegate thinking two")
            await on_text("streaming ")
            await on_text("output")
            return '{"entries":[{"content":"step","priority":"high"}]}', None

        monkeypatch.setattr("isaac.agent.subagents.delegate_tools.stream_with_runner", fake_stream_with_runner)
        monkeypatch.setattr(
            "isaac.agent.subagents.delegate_tools._build_delegate_agent",
            lambda *_args, **_kwargs: object(),
        )

        from isaac.agent.subagents.planner import planner

        result = await planner(SimpleNamespace(tool_call_id="tc-delegate"), task="plan it")
        assert result["error"] is None
    finally:
        reset_delegate_tool_context(ctx)

    thought_updates = [u for u in updates if isinstance(u, AgentThoughtChunk)]
    progress_updates = [u for u in updates if isinstance(u, ToolCallProgress)]
    assert not progress_updates, "Delegate text progress updates should be suppressed"
    assert thought_updates, "Expected delegate tool thought updates"
    thought_text = "".join(update.content.text for update in thought_updates)
    assert "delegate thinking one" in thought_text
    assert "delegate thinking two" in thought_text
