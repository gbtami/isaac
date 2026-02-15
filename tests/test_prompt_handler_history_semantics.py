from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from isaac.agent.brain.prompt_handler import PromptHandler
from isaac.agent.brain.prompt_runner import PromptEnv
from isaac.agent.brain.session_state import SessionState


class _CapturingRunner:
    def __init__(self) -> None:
        self.prompt_texts: list[str] = []

    async def run_stream_events(self, prompt_text: str, **_: object):
        self.prompt_texts.append(prompt_text)

        async def _gen():
            yield "ok"

        return _gen()


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


@pytest.mark.asyncio
async def test_first_turn_stores_raw_user_prompt_not_bootstrap(monkeypatch):
    from isaac.agent import models as model_registry

    monkeypatch.setattr(
        model_registry,
        "list_models",
        lambda: {"codex-model": {"provider": "openai-codex", "model": "gpt-5-codex"}},
    )

    env = _make_env()
    handler = PromptHandler(env)
    runner = _CapturingRunner()
    session_id = "s1"
    handler._sessions[session_id] = SessionState(runner=runner, model_id="codex-model")  # type: ignore[attr-defined]

    await handler.handle_prompt(session_id, "print hello", asyncio.Event())

    state = handler._sessions[session_id]  # type: ignore[attr-defined]
    assert state.history[0]["role"] == "user"
    assert state.history[0]["content"] == "print hello"
    assert runner.prompt_texts
    assert runner.prompt_texts[0] != "print hello"
    assert "you are isaac" in runner.prompt_texts[0].lower()
