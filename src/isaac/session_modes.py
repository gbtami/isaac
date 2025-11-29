"""Session mode helpers mapped to ACP session-mode endpoints.

See: https://agentclientprotocol.com/protocol/session-modes
"""

from __future__ import annotations

from typing import Any, Dict


def available_modes() -> list[dict[str, str]]:
    return [
        {"id": "ask", "name": "Ask", "description": "Normal assistant responses"},
        {"id": "yolo", "name": "Yolo", "description": "Act without extra permission prompts"},
    ]


def build_mode_state(session_modes: Dict[str, str], session_id: str, current_mode: str) -> Any:
    session_modes[session_id] = current_mode
    try:
        from acp.schema import SessionModeState, SessionMode  # type: ignore
    except Exception:
        return None

    modes = available_modes()
    return SessionModeState(
        availableModes=[
            SessionMode(id=m["id"], name=m["name"], description=m["description"]) for m in modes
        ],
        currentModeId=current_mode,
    )
