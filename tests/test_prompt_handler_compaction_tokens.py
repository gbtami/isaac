from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from isaac.agent.brain.session_state import SessionState
from isaac.agent.brain import compaction as compaction_utils
from isaac.agent.brain.prompt_runner import PromptEnv


COMPACT_USER_MAX_TOKENS = 20_000


async def _compact_history(
    env: PromptEnv,
    state: SessionState,
    *,
    session_id: str | None = None,
    ratio: float = 0.5,
    max_history_messages: int = 30,
) -> None:
    await compaction_utils.maybe_compact_history(
        env=env,
        state=state,
        session_id=session_id,
        model_id=state.model_id,
        max_history_messages=max_history_messages,
        auto_compact_ratio=ratio,
        compact_user_message_max_tokens=COMPACT_USER_MAX_TOKENS,
    )


def _make_env() -> tuple[PromptEnv, AsyncMock]:
    noop = AsyncMock()
    notify = AsyncMock()
    env = PromptEnv(
        session_modes={},
        session_last_chunk={},
        send_message_chunk=noop,
        send_thought_chunk=noop,
        send_tool_start=noop,
        send_tool_finish=noop,
        send_plan_update=noop,
        send_notification=notify,
        send_protocol_update=noop,
        request_run_permission=AsyncMock(return_value=True),
        set_usage=lambda *_: None,
    )
    return env, notify


@pytest.mark.asyncio
async def test_prompt_handler_compacts_by_token_limit(monkeypatch):
    async def fake_stream_with_runner(
        _runner,
        _prompt: str,
        *_: object,
        history=None,
        **__: object,
    ):
        _ = history
        return "summary", None

    monkeypatch.setattr("isaac.agent.brain.compaction.stream_with_runner", fake_stream_with_runner)
    monkeypatch.setattr("isaac.agent.brain.compaction.model_registry.get_context_limit", lambda *_: 200)

    env, _ = _make_env()
    state = SessionState(
        runner=object(),
        model_id="m",
        history=[
            {"role": "user", "content": "x" * 500},
            {"role": "assistant", "content": "y" * 500},
            {"role": "assistant", "content": "z" * 10},
        ],
    )

    await _compact_history(env, state, ratio=0.5)

    assert state.history[-1]["role"] == "user"
    assert "summary" in state.history[-1]["content"].lower()
    assert len(state.history) == 2


@pytest.mark.asyncio
async def test_prompt_handler_compacts_from_usage_total(monkeypatch):
    async def fake_stream_with_runner(
        _runner,
        _prompt: str,
        *_: object,
        history=None,
        **__: object,
    ):
        _ = history
        return "summary", None

    monkeypatch.setattr("isaac.agent.brain.compaction.stream_with_runner", fake_stream_with_runner)
    monkeypatch.setattr("isaac.agent.brain.compaction.model_registry.get_context_limit", lambda *_: 200)

    env, _ = _make_env()
    state = SessionState(
        runner=object(),
        model_id="m",
        history=[
            {"role": "user", "content": "short"},
            {"role": "assistant", "content": "done"},
        ],
        last_usage_total_tokens=120,
    )

    await _compact_history(env, state, ratio=0.5)

    assert state.history[-1]["role"] == "user"
    assert "summary" in state.history[-1]["content"].lower()


@pytest.mark.asyncio
async def test_prompt_handler_compaction_notifies_client(monkeypatch):
    async def fake_stream_with_runner(
        _runner,
        _prompt: str,
        *_: object,
        history=None,
        **__: object,
    ):
        _ = history
        return "summary", None

    monkeypatch.setattr("isaac.agent.brain.compaction.stream_with_runner", fake_stream_with_runner)
    monkeypatch.setattr("isaac.agent.brain.compaction.model_registry.get_context_limit", lambda *_: 200)

    env, notify = _make_env()
    state = SessionState(
        runner=object(),
        model_id="m",
        history=[
            {"role": "user", "content": "short"},
            {"role": "assistant", "content": "done"},
        ],
        last_usage_total_tokens=120,
    )

    await _compact_history(env, state, ratio=0.5, session_id="s1")

    assert notify.await_args_list
    call = notify.await_args_list[-1]
    text = call.args[1] if len(call.args) > 1 else call.kwargs.get("message", "")
    assert "Context compacted" in text


@pytest.mark.asyncio
async def test_prompt_handler_compaction_falls_back_on_empty_summary(monkeypatch):
    async def fake_stream_with_runner(
        _runner,
        _prompt: str,
        *_: object,
        history=None,
        **__: object,
    ):
        _ = history
        return "There is no earlier conversation to summarize.", None

    monkeypatch.setattr("isaac.agent.brain.compaction.stream_with_runner", fake_stream_with_runner)
    monkeypatch.setattr("isaac.agent.brain.compaction.model_registry.get_context_limit", lambda *_: 200)

    env, _ = _make_env()
    state = SessionState(
        runner=object(),
        model_id="m",
        history=[
            {"role": "user", "content": "Need tests for demo file"},
            {"role": "assistant", "content": "Updated file tests/test_demo.py [completed]"},
            {"role": "assistant", "content": "done"},
        ],
        last_usage_total_tokens=120,
    )

    await _compact_history(env, state, ratio=0.5, session_id="s1")

    assert state.history[-1]["role"] == "user"
    assert "summary" in state.history[-1]["content"].lower()
    assert "Auto summary" in state.history[-1]["content"]


def test_prompt_handler_compaction_trims_large_messages():
    history = [
        {"role": "assistant", "content": "Prefix\nDiff:\n" + "x" * 2000},
        {"role": "assistant", "content": "Stdout:\n" + "y" * 2000},
        {"role": "assistant", "content": "short"},
    ]

    trimmed = compaction_utils.prepare_compaction_history(history, context_limit=4000)

    assert trimmed
    assert trimmed[-1]["content"] == "short"
    assert "Diff (truncated for compaction):" in trimmed[0]["content"]
    assert len(trimmed[0]["content"]) < 600
    assert "Stdout:" in trimmed[1]["content"]
    assert len(trimmed[1]["content"]) < 900


def test_prompt_handler_compaction_drops_oldest_when_needed():
    history = [
        {"role": "assistant", "content": "a" * 2000},
        {"role": "assistant", "content": "b" * 2000},
        {"role": "assistant", "content": "c" * 2000},
    ]

    trimmed = compaction_utils.prepare_compaction_history(history, context_limit=300)

    assert trimmed
    assert len(trimmed) == 1
    assert trimmed[0]["content"].startswith("c")
