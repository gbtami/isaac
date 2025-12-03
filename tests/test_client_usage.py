from __future__ import annotations

from isaac.client.protocol import _maybe_capture_usage
from isaac.client.session_state import SessionUIState


def test_usage_marker_updates_state():
    state = SessionUIState(current_mode="ask", current_model="model", mcp_servers=[])
    handled = _maybe_capture_usage("[usage] pct=60", state)
    assert handled is True
    assert state.usage_summary == "60% left"


def test_usage_marker_fallback():
    state = SessionUIState(current_mode="ask", current_model="model", mcp_servers=[])
    handled = _maybe_capture_usage("[usage] other=1", state)
    assert handled is True
    assert state.usage_summary is None
