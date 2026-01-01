"""Event payloads emitted by the brain layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


ToolStatus = Literal["in_progress", "completed", "failed"]


@dataclass(frozen=True)
class ToolCallStart:
    """Tool invocation start metadata for UI/protocol adapters."""

    tool_call_id: str
    tool_name: str
    kind: str
    raw_input: dict[str, Any]


@dataclass(frozen=True)
class ToolCallFinish:
    """Tool invocation result metadata for UI/protocol adapters."""

    tool_call_id: str
    tool_name: str
    status: ToolStatus
    raw_output: dict[str, Any]
    old_text: str | None = None
    new_text: str | None = None


__all__ = ["ToolCallFinish", "ToolCallStart", "ToolStatus"]
