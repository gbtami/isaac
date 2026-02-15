from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from isaac.agent.brain import session_ops
from isaac.agent.brain.prompt_runner import PromptEnv
from isaac.agent.brain.session_state import SessionState


def _make_env() -> PromptEnv:
    noop = AsyncMock()
    return PromptEnv(
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


def _factory(*_args, **_kwargs):
    return object()


@pytest.mark.asyncio
async def test_set_session_model_compacts_before_smaller_context_switch(monkeypatch):
    compact_mock = AsyncMock()
    monkeypatch.setattr(session_ops, "maybe_compact_history", compact_mock)
    monkeypatch.setattr(
        session_ops.model_registry,
        "get_context_limit",
        lambda model_id: {"model-old": 1000, "model-new": 200}.get(model_id),
    )

    env = _make_env()
    history = [
        {"role": "user", "content": "request one"},
        {"role": "assistant", "content": "response one"},
    ]
    state = SessionState(
        runner=object(),
        model_id="model-old",
        history=list(history),
        usage_total_tokens_since_compaction=180,
        last_usage_total_tokens=90,
    )

    await session_ops.set_session_model(
        env=env,
        session_id="s1",
        state=state,
        model_id="model-new",
        register_tools=lambda *_: None,
        toolsets=[],
        runner_factory=_factory,
        auto_compact_ratio=0.5,
        compact_user_message_max_tokens=20_000,
        max_history_messages=30,
    )

    compact_mock.assert_awaited_once()
    assert compact_mock.await_args.kwargs["model_id"] == "model-old"
    assert state.model_id == "model-new"
    assert state.history == history


@pytest.mark.asyncio
async def test_set_session_model_skips_compaction_for_larger_context(monkeypatch):
    compact_mock = AsyncMock()
    monkeypatch.setattr(session_ops, "maybe_compact_history", compact_mock)
    monkeypatch.setattr(
        session_ops.model_registry,
        "get_context_limit",
        lambda model_id: {"model-old": 200, "model-new": 1000}.get(model_id),
    )

    env = _make_env()
    history = [{"role": "user", "content": "keep this"}]
    state = SessionState(
        runner=object(),
        model_id="model-old",
        history=list(history),
        usage_total_tokens_since_compaction=500,
        last_usage_total_tokens=200,
    )

    await session_ops.set_session_model(
        env=env,
        session_id="s1",
        state=state,
        model_id="model-new",
        register_tools=lambda *_: None,
        toolsets=[],
        runner_factory=_factory,
        auto_compact_ratio=0.5,
        compact_user_message_max_tokens=20_000,
        max_history_messages=30,
    )

    compact_mock.assert_not_awaited()
    assert state.model_id == "model-new"
    assert state.history == history
