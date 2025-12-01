"""Bridge ACP session history into pydantic-ai history inputs."""

from __future__ import annotations

from typing import Any, Iterable, List

from acp.schema import AgentMessageChunk, UserMessageChunk


def build_chat_history(updates: Iterable[Any]) -> List[dict[str, str]]:
    """Convert ACP session/update notifications into role/content messages."""
    history: list[dict[str, str]] = []
    for update in updates:
        update_obj = getattr(update, "update", update)
        if isinstance(update_obj, UserMessageChunk):
            content = getattr(update_obj, "content", None)
            if content and getattr(content, "text", None):
                history.append({"role": "user", "content": content.text})
        elif isinstance(update_obj, AgentMessageChunk):
            content = getattr(update_obj, "content", None)
            if content and getattr(content, "text", None):
                history.append({"role": "assistant", "content": content.text})
    return history
