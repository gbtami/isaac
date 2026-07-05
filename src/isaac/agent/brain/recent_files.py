"""Recent file tracking and context injection for follow-up prompts."""

from __future__ import annotations

import contextlib
from typing import Any

from pydantic_ai import FunctionToolResultEvent  # type: ignore


def record_recent_file(recent_files: list[str], event: Any, max_recent: int) -> None:
    """Track the most recently edited files for follow-up prompt context."""

    if not isinstance(event, FunctionToolResultEvent):
        return
    result_part = getattr(event, "result", None) or getattr(event, "part", None)
    tool_name = getattr(result_part, "tool_name", None) or ""
    if tool_name not in {"edit_file", "apply_patch"}:
        return
    content = getattr(result_part, "content", None)
    path = None
    if isinstance(content, dict):
        path = content.get("path")
    if not path:
        return
    normalized = str(path).strip()
    if not normalized:
        return
    with contextlib.suppress(ValueError):
        recent_files.remove(normalized)
    recent_files.append(normalized)
    if len(recent_files) > max_recent:
        del recent_files[:-max_recent]


def recent_files_context_text(recent_files: list[str], context_count: int) -> str | None:
    """Return the runtime hint used for ambiguous file follow-ups."""

    if not recent_files:
        return None
    recent = recent_files[-context_count:]
    if not recent:
        return None
    return (
        "Recent files touched (most recent last): "
        f"{', '.join(recent)}.\n"
        "If the user refers to an unspecified file, assume they mean the most recent file above."
    )
