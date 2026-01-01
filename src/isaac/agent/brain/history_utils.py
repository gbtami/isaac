"""History helpers for prompt handling."""

from __future__ import annotations

from typing import Any

from isaac.agent.history_types import ChatMessage


def extract_usage_total(usage: Any) -> int | None:
    """Return total token usage from a RunUsage-like object or dict."""

    if usage is None:
        return None

    def _get(field: str) -> int | None:
        if hasattr(usage, field):
            value = getattr(usage, field)
            return int(value) if isinstance(value, (int, float)) else None
        if isinstance(usage, dict) and field in usage:
            value = usage.get(field)
            return int(value) if isinstance(value, (int, float)) else None
        return None

    total = _get("total_tokens")
    if total is not None:
        return total
    input_tokens = _get("input_tokens") or _get("prompt_tokens")
    output_tokens = _get("output_tokens") or _get("completion_tokens")
    if input_tokens is not None and output_tokens is not None:
        return input_tokens + output_tokens
    return None


def trim_history(history: list[ChatMessage], limit: int) -> list[ChatMessage]:
    """Return the most recent history up to the limit."""

    if limit <= 0:
        return []
    if len(history) <= limit:
        return list(history)
    return list(history[-limit:])
