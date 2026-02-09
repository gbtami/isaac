from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from isaac.client.slash import _handle_mode, _handle_model
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
