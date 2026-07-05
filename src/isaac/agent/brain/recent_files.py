"""Recent file tracking and context injection for follow-up prompts."""

from __future__ import annotations

import contextlib
from typing import Any

from isaac.agent.history_types import ChatMessage

try:
    from pydantic_ai import FunctionToolResultEvent  # type: ignore
except ImportError:  # pragma: no cover - older pydantic-ai compatibility
    from pydantic_ai.messages import FunctionToolResultEvent  # type: ignore


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


def inject_recent_files_context(
    history: list[ChatMessage], recent_files: list[str], context_count: int
) -> list[ChatMessage]:
    """Add recent file hints to the model context for ambiguous follow-ups.

    This helper is retained for compatibility with older tests and embeddings.
    Normal prompt handling now contributes this hint as a per-run Pydantic AI
    capability instead of mutating the chat history sent through ACP/session
    persistence.
    """

    message = recent_files_context_text(recent_files, context_count)
    if message is None:
        return list(history)
    return [*history, {"role": "system", "content": message}]
