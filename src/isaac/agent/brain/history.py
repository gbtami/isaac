"""Bridge ACP session history into pydantic-ai history inputs."""

from __future__ import annotations

from typing import Any, Iterable, List

from acp.schema import AgentMessageChunk, AgentPlanUpdate, ToolCallProgress, UserMessageChunk


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
        elif isinstance(update_obj, ToolCallProgress):
            blocks = getattr(update_obj, "content", None) or []
            for block in blocks:
                inner = getattr(block, "content", None)
                text = getattr(inner, "text", None) if inner else None
                if text:
                    history.append({"role": "assistant", "content": text})
        elif isinstance(update_obj, AgentPlanUpdate):
            entries = getattr(update_obj, "entries", None) or []
            text = "\n".join(
                f"- {getattr(e, 'content', '')}" for e in entries if getattr(e, "content", "")
            )
            if text:
                history.append({"role": "assistant", "content": f"Plan:\n{text}"})
    return history
