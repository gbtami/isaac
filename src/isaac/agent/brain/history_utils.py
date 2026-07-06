"""History helpers for prompt handling and context selection."""

from __future__ import annotations

import re
from typing import Any

from isaac.agent.brain.compaction import approx_text_tokens, is_summary_message, truncate_text_tokens
from isaac.agent.history_types import ChatMessage

_DEFAULT_UNKNOWN_CONTEXT_LIMIT = 64_000
_DEFAULT_HISTORY_RATIO = 0.55
_DEFAULT_RESERVED_PROMPT_TOKENS = 8_192
_MIN_HISTORY_BUDGET_TOKENS = 2_000
_MAX_MESSAGE_TOKENS = 2_000
_RECENT_CONTEXT_MESSAGES = 16
_MAX_SELECTED_MESSAGES = 120
_ALWAYS_KEEP_TOOL_NAMES = {
    "edit_file",
    "apply_patch",
    "run_command",
    "delegate",
    "planner",
    "reviewer",
    "coding",
}
_DISCOVERY_TOOL_NAMES = {"read_file", "list_files", "file_summary", "code_search", "fetch_url"}
_STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "because",
    "before",
    "could",
    "from",
    "have",
    "into",
    "just",
    "more",
    "need",
    "next",
    "please",
    "should",
    "that",
    "the",
    "their",
    "then",
    "there",
    "this",
    "with",
    "would",
    "your",
}


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
    """Return the most recent history up to the limit.

    Kept for compaction/model-switch callers that need a simple recency slice.
    Prompt assembly should prefer :func:`select_context_history`.
    """

    if limit <= 0:
        return []
    if len(history) <= limit:
        return list(history)
    return list(history[-limit:])


def context_history_token_budget(
    context_limit: int | None,
    *,
    ratio: float = _DEFAULT_HISTORY_RATIO,
    reserved_prompt_tokens: int = _DEFAULT_RESERVED_PROMPT_TOKENS,
) -> int:
    """Return the approximate token budget available for prior-turn history."""

    limit = context_limit if context_limit and context_limit > 0 else _DEFAULT_UNKNOWN_CONTEXT_LIMIT
    ratio_budget = max(1, int(limit * ratio))
    reserved_budget = limit - reserved_prompt_tokens
    if reserved_budget <= 0:
        reserved_budget = max(1, int(limit * 0.45))
    return max(_MIN_HISTORY_BUDGET_TOKENS, min(ratio_budget, reserved_budget))


def select_context_history(
    history: list[ChatMessage],
    *,
    current_prompt: str = "",
    context_limit: int | None = None,
    recent_messages: int = _RECENT_CONTEXT_MESSAGES,
    max_messages: int = _MAX_SELECTED_MESSAGES,
    max_message_tokens: int = _MAX_MESSAGE_TOKENS,
) -> list[ChatMessage]:
    """Select prompt history by relevance and budget instead of fixed recency.

    The selector keeps the recent conversational tail for coherence, but also
    preserves older coding-critical facts such as compaction summaries, file
    edits, command/test results, delegate outputs, and relevant read/search
    observations. This makes follow-up prompts less dependent on the last N raw
    messages while keeping provider-visible history plain user/assistant text.
    """

    normalized = _normalize_history(history, max_message_tokens=max_message_tokens)
    if not normalized or max_messages <= 0:
        return []

    budget = context_history_token_budget(context_limit)
    prompt_terms = _prompt_terms(current_prompt)
    scored: list[tuple[int, int, ChatMessage]] = []
    last_index = len(normalized) - 1
    recent_start = max(0, len(normalized) - max(recent_messages, 0))

    for idx, msg in enumerate(normalized):
        priority = _message_priority(
            msg,
            idx=idx,
            last_index=last_index,
            recent_start=recent_start,
            prompt_terms=prompt_terms,
        )
        if priority <= 0:
            continue
        # Sort by priority, then recency. We restore chronological order below.
        scored.append((priority, idx, msg))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected_indexes: set[int] = set()
    selected_tokens = 0
    for _priority, idx, msg in scored:
        if len(selected_indexes) >= max_messages:
            break
        tokens = approx_text_tokens(str(msg.get("content") or "")) + 4
        if selected_indexes and selected_tokens + tokens > budget:
            continue
        selected_indexes.add(idx)
        selected_tokens += tokens

    # Always return messages in original order. Pydantic AI expects normal
    # conversational order even when the set was selected by priority.
    return [normalized[idx] for idx in sorted(selected_indexes)]


def _normalize_history(history: list[ChatMessage], *, max_message_tokens: int) -> list[ChatMessage]:
    normalized: list[ChatMessage] = []
    for msg in history:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "").strip()
        content = str(msg.get("content") or "").strip()
        if role not in {"user", "assistant", "system"} or not content:
            continue
        if role == "user" and is_summary_message(content):
            # Summary blocks are already compact. Keep them intact unless they
            # are extreme due to a malformed checkpoint.
            trimmed = truncate_text_tokens(content, max(max_message_tokens * 2, max_message_tokens))
        else:
            trimmed = truncate_text_tokens(content, max_message_tokens)
        copied: ChatMessage = {"role": role, "content": trimmed}
        for key in ("source", "synthetic", "checkpoint", "tool_name", "tool_kind"):
            if key in msg:
                copied[key] = msg[key]
        normalized.append(copied)
    return normalized


def _message_priority(
    msg: ChatMessage,
    *,
    idx: int,
    last_index: int,
    recent_start: int,
    prompt_terms: set[str],
) -> int:
    role = str(msg.get("role") or "")
    source = str(msg.get("source") or "")
    tool_name = str(msg.get("tool_name") or "").lower()
    tool_kind = str(msg.get("tool_kind") or "").lower()
    content = str(msg.get("content") or "")
    is_recent = idx >= recent_start
    is_last = idx == last_index
    relevant = _matches_terms(content, prompt_terms)

    if bool(msg.get("checkpoint")) or source == "compaction_summary" or is_summary_message(content):
        return 95
    if source == "tool_summary":
        if tool_name in _ALWAYS_KEEP_TOOL_NAMES or tool_kind in {"edit", "execute", "think"}:
            return 90 if is_recent else 82
        if tool_name in _DISCOVERY_TOOL_NAMES or tool_kind in {"read", "search", "fetch"}:
            if relevant or is_recent:
                return 76 if relevant else 58
            return 25
        return 50 if is_recent else 25
    if is_last:
        return 88
    if is_recent:
        return 72 if role in {"user", "assistant"} else 60
    if relevant and role in {"user", "assistant"}:
        return 55
    return 0


def _prompt_terms(text: str) -> set[str]:
    terms: set[str] = set()
    for match in re.finditer(r"[A-Za-z0-9_./:+-]{3,}", text.lower()):
        term = match.group(0).strip(".,:;()[]{}<>`'\"")
        if not term or term in _STOPWORDS:
            continue
        terms.add(term)
        if "/" in term:
            terms.update(part for part in term.split("/") if len(part) >= 3 and part not in _STOPWORDS)
        if "." in term:
            terms.update(part for part in term.split(".") if len(part) >= 3 and part not in _STOPWORDS)
    return terms


def _matches_terms(text: str, terms: set[str]) -> bool:
    if not terms:
        return False
    lowered = text.lower()
    return any(term in lowered for term in terms)


__all__ = [
    "context_history_token_budget",
    "extract_usage_total",
    "select_context_history",
    "trim_history",
]
