"""Bridge ACP session history into pydantic-ai history inputs."""

from __future__ import annotations

from typing import Any, Iterable, List

from acp.schema import AgentMessageChunk, AgentPlanUpdate, ToolCallProgress, UserMessageChunk


def build_chat_history(updates: Iterable[Any]) -> List[dict[str, str]]:
    """Convert ACP session/update notifications into role/content messages."""
    history: list[dict[str, str]] = []
    skip_next_assistant = False
    for update in updates:
        update_obj = getattr(update, "update", update)
        if isinstance(update_obj, UserMessageChunk):
            content = getattr(update_obj, "content", None)
            if content and getattr(content, "text", None):
                text = content.text
                if text.startswith("/"):
                    skip_next_assistant = True
                    continue
                history.append({"role": "user", "content": text, "_src": "user"})
        elif isinstance(update_obj, AgentMessageChunk):
            if skip_next_assistant:
                skip_next_assistant = False
                continue
            content = getattr(update_obj, "content", None)
            if content and getattr(content, "text", None):
                history.append({"role": "assistant", "content": content.text, "_src": "agent_chunk"})
        elif isinstance(update_obj, ToolCallProgress):
            blocks = getattr(update_obj, "content", None) or []
            for block in blocks:
                inner = getattr(block, "content", None)
                text = getattr(inner, "text", None) if inner else None
                if text:
                    history.append({"role": "assistant", "content": text, "_src": "tool"})
        elif isinstance(update_obj, AgentPlanUpdate):
            entries = getattr(update_obj, "entries", None) or []
            text = "\n".join(f"- {getattr(e, 'content', '')}" for e in entries if getattr(e, "content", ""))
            if text:
                history.append({"role": "assistant", "content": f"Plan:\n{text}", "_src": "plan"})
    # Merge sequences from the same role (assistant/user) to avoid fragmented context.
    merged: list[dict[str, str]] = []
    current_role: str | None = None
    current_src: str | None = None
    buffer: list[str] = []
    for msg in history:
        role = msg.get("role")
        src = msg.get("_src")
        content = msg.get("content", "")
        if role != current_role:
            if current_role is not None:
                merged.append({"role": current_role, "content": "".join(buffer)})
            current_role = role
            current_src = src
            buffer = [content]
        else:
            if current_src != src and buffer:
                buffer.append("\n")
            buffer.append(content)
            current_src = src
    if current_role is not None:
        merged.append({"role": current_role, "content": "".join(buffer)})
    for m in merged:
        m.pop("_src", None)
    return merged
