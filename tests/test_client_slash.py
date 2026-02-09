from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from acp import text_block

from isaac.client.slash import _handle_mode, _handle_model, _handle_status
from isaac.client.session_state import SessionUIState


@pytest.mark.asyncio
async def test_model_command_uses_picker_when_argument_missing() -> None:
    conn = AsyncMock()
    conn.set_session_config_option = AsyncMock(return_value={"config_options": []})
    state = SessionUIState(
        current_mode="ask",
        current_model="openai:gpt-5",
        mcp_servers=[],
        config_option_ids={"model": "model"},
        config_option_values={"model": {"openai:gpt-5", "openai:gpt-5-mini"}},
    )

    async def _pick(_label: str, _options: list[str], _current: str | None) -> str | None:
        return "openai:gpt-5-mini"

    state.select_option = _pick

    handled = await _handle_model(conn, "session-1", state, lambda: None, "")

    assert handled is True
    conn.set_session_config_option.assert_awaited_once_with(
        config_id="model",
        session_id="session-1",
        value="openai:gpt-5-mini",
    )
    assert state.current_model == "openai:gpt-5-mini"


@pytest.mark.asyncio
async def test_mode_command_uses_picker_when_argument_missing() -> None:
    conn = AsyncMock()
    conn.set_session_config_option = AsyncMock(return_value={"config_options": []})
    state = SessionUIState(
        current_mode="ask",
        current_model="openai:gpt-5",
        mcp_servers=[],
        config_option_ids={"mode": "mode"},
        config_option_values={"mode": {"ask", "yolo"}},
    )
    reset = Mock()

    async def _pick(_label: str, _options: list[str], _current: str | None) -> str | None:
        return "yolo"

    state.select_option = _pick

    handled = await _handle_mode(conn, "session-2", state, reset, "")

    assert handled is True
    conn.set_session_config_option.assert_awaited_once_with(
        config_id="mode",
        session_id="session-2",
        value="yolo",
    )
    assert state.current_mode == "yolo"
    reset.assert_called_once_with()


@pytest.mark.asyncio
async def test_status_refreshes_usage_silently(capsys: pytest.CaptureFixture[str]) -> None:
    conn = AsyncMock()
    state = SessionUIState(
        current_mode="ask",
        current_model="openai:gpt-5",
        mcp_servers=[],
        session_id="session-3",
        cwd="/tmp/project",
        usage_summary="Usage: old usage",
    )

    async def _fake_prompt(*, prompt, session_id):
        assert state.suppress_usage_output is True
        assert session_id == "session-3"
        assert len(prompt) == 1
        assert getattr(prompt[0], "text", "") == getattr(text_block("/usage"), "text", "")
        state.usage_summary = "Usage: refreshed usage"

    conn.prompt = AsyncMock(side_effect=_fake_prompt)

    handled = await _handle_status(conn, "session-3", state, lambda: None, "")

    assert handled is True
    assert state.suppress_usage_output is False
    out = capsys.readouterr().out
    assert "Usage: refreshed usage" in out


@pytest.mark.asyncio
async def test_status_restores_suppression_when_refresh_fails(capsys: pytest.CaptureFixture[str]) -> None:
    conn = AsyncMock()
    conn.prompt = AsyncMock(side_effect=RuntimeError("boom"))
    state = SessionUIState(
        current_mode="ask",
        current_model="openai:gpt-5",
        mcp_servers=[],
        session_id="session-4",
        cwd="/tmp/project",
        usage_summary="Usage: cached usage",
    )

    handled = await _handle_status(conn, "session-4", state, lambda: None, "")

    assert handled is True
    assert state.suppress_usage_output is False
    out = capsys.readouterr().out
    assert "Usage: cached usage" in out
