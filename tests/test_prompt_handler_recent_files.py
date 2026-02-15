from __future__ import annotations

import asyncio

from unittest.mock import AsyncMock

import pytest

from isaac.agent.brain.prompt_handler import PromptHandler
from isaac.agent.brain.session_state import SessionState
from isaac.agent.brain.prompt_runner import PromptEnv


@pytest.mark.asyncio
async def test_prompt_handler_injects_recent_files_context(monkeypatch):
    captured: dict[str, object] = {}

    async def fake_stream_with_runner(
        _runner,
        _prompt: str,
        *_: object,
        history=None,
        **__: object,
    ):
        captured["history"] = history
        return "ok", None

    monkeypatch.setattr("isaac.agent.brain.prompt_handler.stream_with_runner", fake_stream_with_runner)

    noop = AsyncMock()
    env = PromptEnv(
        session_modes={},
        session_cwds={},
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
    handler = PromptHandler(env, register_tools=lambda *_: None)
    handler._sessions["s"] = SessionState(
        runner=object(),
        model_id="m",
        history=[{"role": "assistant", "content": "done"}],
        recent_files=["alpha.py", "beta.py"],
    )

    await handler.handle_prompt("s", "do thing", asyncio.Event())

    history = captured.get("history")
    assert isinstance(history, list)
    assert any(
        msg.get("role") == "system" and "Recent files touched" in str(msg.get("content"))
        for msg in history
        if isinstance(msg, dict)
    )
