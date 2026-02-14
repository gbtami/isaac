"""History processor helpers for pydantic-ai runners."""

from __future__ import annotations

from typing import Any, Sequence

from pydantic_ai import messages as ai_messages  # type: ignore


def _clean_content_text(text: Any) -> str | None:
    if text is None:
        return None
    cleaned = str(text).strip()
    return cleaned or None


async def sanitize_message_history(
    ctx_or_messages: Any,
    messages: Sequence[ai_messages.ModelMessage] | None = None,
) -> list[ai_messages.ModelMessage]:
    """Drop empty text/system/user parts to reduce provider-side 400 errors."""

    if messages is None:
        raw_messages = ctx_or_messages
    else:
        raw_messages = messages

    if raw_messages is None:
        return []

    cleaned_messages: list[ai_messages.ModelMessage] = []
    for message in raw_messages:
        if isinstance(message, ai_messages.ModelRequest):
            cleaned_parts: list[Any] = []
            for part in message.parts:
                if isinstance(part, ai_messages.SystemPromptPart):
                    cleaned = _clean_content_text(part.content)
                    if cleaned:
                        cleaned_parts.append(ai_messages.SystemPromptPart(content=cleaned))
                    continue
                if isinstance(part, ai_messages.UserPromptPart):
                    cleaned = _clean_content_text(part.content)
                    if cleaned:
                        cleaned_parts.append(ai_messages.UserPromptPart(content=cleaned))
                    continue
                cleaned_parts.append(part)
            if cleaned_parts:
                cleaned_messages.append(ai_messages.ModelRequest(parts=cleaned_parts))
            continue
        if isinstance(message, ai_messages.ModelResponse):
            cleaned_parts = []
            for part in message.parts:
                if isinstance(part, ai_messages.TextPart):
                    cleaned = _clean_content_text(part.content)
                    if cleaned:
                        cleaned_parts.append(ai_messages.TextPart(content=cleaned))
                    continue
                cleaned_parts.append(part)
            if cleaned_parts:
                cleaned_messages.append(ai_messages.ModelResponse(parts=cleaned_parts))
            continue
        cleaned_messages.append(message)
    return cleaned_messages
