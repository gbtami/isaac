"""Lightweight UI/session state shared across client components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from isaac.client.display import ThinkingStatus


@dataclass
class SessionUIState:
    current_mode: str
    current_model: str
    mcp_servers: list[str]
    collect_models: bool = False
    model_buffer: list[str] | None = None
    show_welcome_on_start: bool = True
    show_thinking: bool = True
    cancel_requested: bool = False
    usage_summary: str | None = None
    available_agent_commands: dict[str, str] = field(default_factory=dict)
    local_slash_commands: set[str] = field(default_factory=set)
    config_option_ids: dict[str, str] = field(default_factory=dict)
    config_option_values: dict[str, set[str]] = field(default_factory=dict)
    session_id: str | None = None
    cwd: str | None = None
    refresh_ui: Callable[[], None] | None = None
    thinking_status: "ThinkingStatus | None" = None

    def notify_changed(self) -> None:
        """Ask the UI to redraw when status fields change."""
        if self.refresh_ui is None:
            return
        try:
            self.refresh_ui()
        except Exception:
            return
