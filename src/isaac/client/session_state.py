"""Lightweight UI/session state shared across client components."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SessionUIState:
    current_mode: str
    current_model: str
    mcp_servers: list[str]
    prompt_strategy: str = "handoff"
    collect_models: bool = False
    model_buffer: list[str] | None = None
    show_status_on_start: bool = True
    pending_newline: bool = False
    show_thinking: bool = False
    cancel_requested: bool = False
    usage_summary: str | None = None
    available_agent_commands: dict[str, str] = field(default_factory=dict)
