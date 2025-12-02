"""Session-scoped thinking/debug toggle."""

from __future__ import annotations

from isaac.client.session_state import SessionUIState


def toggle_thinking(state: SessionUIState, value: bool) -> str:
    state.show_thinking = value
    return "Thinking output enabled." if value else "Thinking output disabled."
