"""Helpers for token-aware history compaction and summary handling."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, TYPE_CHECKING

from isaac.agent import models as model_registry
from isaac.agent.history_types import ChatMessage
from isaac.agent.runner import stream_with_runner
from isaac.log_utils import log_context as log_ctx, log_event

if TYPE_CHECKING:
    from isaac.agent.brain.session_state import SessionState

logger = logging.getLogger(__name__)

COMPACT_PROMPT = (
    "You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary for another LLM "
    "that will resume the task.\n\n"
    "Include:\n"
    "- Current progress and key decisions made\n"
    "- Important context, constraints, or user preferences\n"
    "- What remains to be done (clear next steps)\n"
    "- Any critical data, examples, or references needed to continue\n\n"
    "Be concise, structured, and focused on helping the next LLM seamlessly continue the work."
)
SUMMARY_PREFIX = (
    "Another language model started to solve this problem and produced a summary of its thinking "
    "process. You also have access to the state of the tools that were used by that language model. "
    "Use this to build on the work that has already been done and avoid duplicating work. "
    "Here is the summary produced by the other language model, use the information in this summary "
    "to assist with your own analysis:"
)

_COMPACTION_HISTORY_RATIO = 0.8


def auto_compact_token_limit(context_limit: int | None, ratio: float) -> int | None:
    """Return the token threshold that triggers history compaction."""

    if not context_limit or context_limit <= 0:
        return None
    limit = int(context_limit * ratio)
    return max(limit, 1)


def estimate_history_tokens(history: list[ChatMessage]) -> int:
    """Estimate token usage from history content (roughly 4 bytes/token)."""

    total = 0
    for msg in history:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if content is None:
            continue
        text = str(content)
        total += approx_text_tokens(text)
        total += 4  # small overhead per message
    return total


def prepare_compaction_history(history: list[ChatMessage], context_limit: int | None) -> list[ChatMessage]:
    """Shrink history content for compaction prompts to avoid context blow-ups."""

    trimmed: list[ChatMessage] = []
    for msg in history:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if not role or content is None:
            continue
        compacted = str(content)
        if role == "user" and is_summary_message(compacted):
            continue
        compacted = compact_message_text(compacted)
        trimmed.append({"role": str(role), "content": compacted})

    if not context_limit or context_limit <= 0:
        return trimmed

    max_tokens = int(context_limit * _COMPACTION_HISTORY_RATIO)
    while trimmed and estimate_history_tokens(trimmed) > max_tokens:
        trimmed.pop(0)
    return trimmed


def compact_message_text(text: str, limit: int = 800) -> str:
    """Trim large tool summaries before feeding them into the compaction prompt."""

    normalized = text.strip()
    if not normalized:
        return normalized

    def _truncate(value: str, max_len: int) -> str:
        if len(value) <= max_len:
            return value
        return value[:max_len].rstrip() + "..."

    if "Diff:\n" in normalized:
        header, diff = normalized.split("Diff:\n", 1)
        return "\n".join(
            [
                header.strip(),
                "Diff (truncated for compaction):",
                _truncate(diff.strip(), 400),
            ]
        )

    if "Stdout:\n" in normalized or "Stderr:\n" in normalized:
        lines = []
        for chunk in normalized.splitlines():
            if chunk.startswith("Stdout:"):
                lines.append(chunk)
            elif chunk.startswith("Stderr:"):
                lines.append(chunk)
            else:
                lines.append(chunk)
        return _truncate("\n".join(lines), limit)

    return _truncate(normalized, limit)


def summary_rejected(summary: str) -> bool:
    """Detect unusable compaction summaries returned by the model."""

    if not summary:
        return True
    lowered = summary.lower()
    return any(
        phrase in lowered
        for phrase in (
            "no earlier conversation",
            "first turn",
            "please provide the task",
            "no conversation to summarize",
        )
    )


def fallback_summary(history: list[ChatMessage]) -> str:
    """Build a simple summary from existing history when the model fails."""

    lines: list[str] = []
    last_user = None
    last_assistant = None
    for msg in history:
        if msg.get("role") == "user":
            last_user = str(msg.get("content") or "").strip()
        elif msg.get("role") == "assistant":
            last_assistant = str(msg.get("content") or "").strip()

    if last_user:
        lines.append(f"User request: {last_user[:200]}")

    for msg in history:
        content = str(msg.get("content") or "").strip()
        if content.startswith("Updated file "):
            lines.append(content.splitlines()[0])
        elif content.startswith("Ran command:"):
            lines.append(content.splitlines()[0])
        elif content.startswith("Delegated to "):
            lines.append(content.splitlines()[0])

    if last_assistant:
        lines.append(f"Last assistant response: {last_assistant[:200]}")

    if not lines:
        return "Summary unavailable; conversation was compacted due to context limits."

    unique_lines: list[str] = []
    for line in lines:
        if line not in unique_lines:
            unique_lines.append(line)
    return "Auto summary:\n- " + "\n- ".join(unique_lines[:8])


def is_summary_message(text: str) -> bool:
    """Return True when the message is a compaction summary."""

    summary_text = text.strip()
    return summary_text.startswith(SUMMARY_PREFIX)


def collect_user_messages(history: list[ChatMessage]) -> list[str]:
    """Collect non-summary user messages for compacted history rebuild."""

    messages: list[str] = []
    for msg in history:
        if msg.get("role") != "user":
            continue
        content = str(msg.get("content") or "").strip()
        if not content or is_summary_message(content):
            continue
        messages.append(content)
    return messages


def build_compacted_history(user_messages: list[str], summary_text: str, max_tokens: int) -> list[ChatMessage]:
    """Rebuild history as recent user messages followed by a summary block."""

    selected: list[str] = []
    remaining = max_tokens
    if remaining > 0:
        for message in reversed(user_messages):
            if remaining <= 0:
                break
            tokens = approx_text_tokens(message)
            if tokens <= remaining:
                selected.append(message)
                remaining -= tokens
            else:
                selected.append(truncate_text_tokens(message, remaining))
                break
        selected.reverse()

    history: list[ChatMessage] = [{"role": "user", "content": message} for message in selected if message.strip()]

    summary = summary_text.strip() if summary_text else ""
    if not summary:
        summary = "(no summary available)"
    if not summary.startswith(SUMMARY_PREFIX):
        summary = f"{SUMMARY_PREFIX}\n{summary}"
    history.append({"role": "user", "content": summary})
    return history


def approx_text_tokens(text: str) -> int:
    """Estimate tokens from byte length (roughly 4 bytes per token)."""

    byte_len = len(text.encode("utf-8"))
    return max(1, (byte_len + 3) // 4)


def truncate_text_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to an approximate token budget with a marker."""

    if max_tokens <= 0:
        return ""
    max_bytes = max_tokens * 4
    data = text.encode("utf-8")
    if len(data) <= max_bytes:
        return text
    trimmed = data[:max_bytes].decode("utf-8", errors="ignore").rstrip()
    return f"{trimmed}... [truncated]"


async def maybe_compact_history(
    *,
    env: Any,
    state: "SessionState",
    session_id: str | None,
    model_id: str | None,
    max_history_messages: int,
    auto_compact_ratio: float,
    compact_user_message_max_tokens: int,
) -> None:
    """Compact older history into a summary when it grows too large."""

    runner = state.runner
    if runner is None:
        return

    context_limit = model_registry.get_context_limit(model_id or "")
    token_limit = auto_compact_token_limit(context_limit, auto_compact_ratio)
    if token_limit is None:
        if len(state.history) <= max_history_messages:
            return
    else:
        estimated = estimate_history_tokens(state.history)
        usage_total = state.last_usage_total_tokens
        if usage_total is not None:
            estimated = max(estimated, usage_total)
        if estimated <= token_limit:
            return
        with log_ctx(session_id=session_id, model_id=model_id):
            log_event(
                logger,
                "prompt.history.compact",
                level=logging.DEBUG,
                estimated_tokens=estimated,
                usage_total_tokens=state.last_usage_total_tokens,
                token_limit=token_limit,
                context_limit=context_limit,
            )

    if not state.history:
        return

    async def _noop(_: str) -> None:
        return None

    summary_text = None
    try:
        compaction_history = prepare_compaction_history(state.history, context_limit)
        summary_text, _ = await stream_with_runner(
            runner,
            COMPACT_PROMPT,
            _noop,
            _noop,
            asyncio.Event(),
            history=compaction_history,
            log_context="history_compact",
        )
    except Exception:
        summary_text = None

    summary_content = (summary_text or "").strip()
    if summary_rejected(summary_content):
        summary_content = fallback_summary(state.history)
        with log_ctx(session_id=session_id, model_id=model_id):
            log_event(logger, "prompt.history.compact.fallback", level=logging.DEBUG)
    user_messages = collect_user_messages(state.history)
    state.history = build_compacted_history(user_messages, summary_content, compact_user_message_max_tokens)
    if session_id:
        await env.send_notification(
            session_id,
            "Context compacted to stay within model limits. Recent details may be summarized.",
        )
