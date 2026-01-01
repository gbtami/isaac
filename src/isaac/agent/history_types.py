"""Shared history typing helpers."""

from __future__ import annotations

from typing import Sequence, TypedDict, TypeAlias

from pydantic_ai import messages as ai_messages  # type: ignore


class ChatMessage(TypedDict):
    """Simple chat history message for role/content tracking."""

    role: str
    content: str


HistoryInput: TypeAlias = Sequence[ChatMessage] | Sequence[ai_messages.ModelMessage]

__all__ = ["ChatMessage", "HistoryInput"]
