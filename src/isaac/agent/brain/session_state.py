"""Session state container for prompt handling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from isaac.agent.history_types import ChatMessage


@dataclass
class SessionState:
    """State for sessions using the prompt handler."""

    runner: Any | None = None
    model_id: str | None = None
    history: list[ChatMessage] = field(default_factory=list)
    recent_files: list[str] = field(default_factory=list)
    last_usage_total_tokens: int | None = None
    model_error: str | None = None
    model_error_notified: bool = False
