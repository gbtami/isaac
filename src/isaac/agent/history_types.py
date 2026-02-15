"""Shared history typing helpers."""

from __future__ import annotations

from typing import Any, Sequence, TypedDict, TypeAlias

from pydantic_ai import messages as ai_messages  # type: ignore


class ChatMessage(TypedDict, total=False):
    """Chat history message with optional metadata for compaction/replay decisions.

    `role`/`content` are still the model-visible fields.
    Additional optional keys let us keep structured context bookkeeping without
    affecting provider payloads (the runner only reads role/content).
    """

    role: str
    content: str
    source: str
    synthetic: bool
    checkpoint: dict[str, Any]


HistoryInput: TypeAlias = Sequence[ChatMessage] | Sequence[ai_messages.ModelMessage]

__all__ = ["ChatMessage", "HistoryInput"]
