"""Usage reporting helpers."""

from __future__ import annotations

from typing import Any


def normalize_usage(usage: Any) -> Any:
    """Resolve usage to a value if providers expose it as a callable."""

    if usage is None:
        return None
    if callable(usage):
        try:
            return usage()
        except Exception:
            return None
    return usage


def format_usage_summary(usage: Any, context_limit: int | None, model_id: str) -> str:
    """Human-friendly usage summary for on-demand display."""

    if usage is None:
        return "Usage not available for the last run (provider may not return token data)."

    def _get(field: str) -> int | None:
        if hasattr(usage, field):
            val = getattr(usage, field)
            return int(val) if isinstance(val, (int, float)) else None
        if isinstance(usage, dict) and field in usage:
            val = usage.get(field)
            return int(val) if isinstance(val, (int, float)) else None
        return None

    input_tokens = _get("input_tokens") or _get("prompt_tokens")
    output_tokens = _get("output_tokens") or _get("completion_tokens")
    total_tokens = _get("total_tokens") or (
        input_tokens + output_tokens
        if input_tokens is not None and output_tokens is not None
        else None
    )

    parts: list[str] = []
    if input_tokens is not None:
        parts.append(f"input={input_tokens}")
    if output_tokens is not None:
        parts.append(f"output={output_tokens}")
    if total_tokens is not None:
        parts.append(f"total={total_tokens}")

    remaining_txt = ""
    if context_limit is not None and input_tokens is not None:
        remaining = max(0, context_limit - input_tokens)
        pct_left = max(0.0, remaining / context_limit * 100.0)
        remaining_txt = f"context remaining ~{pct_left:.0f}% of {context_limit} tokens (model={model_id or 'unknown'})"

    if not parts and not remaining_txt:
        return "Usage data unavailable for the last run."

    if remaining_txt:
        parts.append(remaining_txt)
    return "Usage: " + ", ".join(parts)
