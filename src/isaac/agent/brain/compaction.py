"""Helpers for token-aware history compaction and summary handling."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel, Field, ValidationError

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
    "Return JSON only with this exact schema:\n"
    "{\n"
    '  "progress": [str],\n'
    '  "key_decisions": [str],\n'
    '  "user_preferences": [str],\n'
    '  "remaining_steps": [str],\n'
    '  "critical_artifacts": [str],\n'
    '  "risks": [str]\n'
    "}\n\n"
    "Keep arrays concise and practical; prefer concrete file names, commands, and next steps."
)
SUMMARY_PREFIX = (
    "Another language model started to solve this problem and produced a summary of its thinking "
    "process. You also have access to the state of the tools that were used by that language model. "
    "Use this to build on the work that has already been done and avoid duplicating work. "
    "Here is the summary produced by the other language model, use the information in this summary "
    "to assist with your own analysis:"
)

_COMPACTION_HISTORY_RATIO = 0.8
_COMPACTION_MAX_OVERFLOW_RETRIES = 8
_CONTEXT_OVERFLOW_ERROR_HINTS = (
    "context",
    "token",
    "maximum",
    "too long",
    "length",
    "overflow",
    "window",
    "exceed",
    "input is too large",
)
_SYNTHETIC_BOOTSTRAP_HINTS = (
    "you are a coding agent running in the codex cli",
    "within this context, codex refers to the open-source agentic coding interface",
    "## responsiveness",
    "<identity>",
    "you are antigravity",
    "<web_application_development>",
)


class CompactionCheckpoint(BaseModel):
    """Structured checkpoint payload retained after compaction.

    We keep this schema intentionally small and task-focused so the next turn gets
    concrete, model-useful continuity rather than verbose narrative text.
    """

    progress: list[str] = Field(default_factory=list)
    key_decisions: list[str] = Field(default_factory=list)
    user_preferences: list[str] = Field(default_factory=list)
    remaining_steps: list[str] = Field(default_factory=list)
    critical_artifacts: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


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
        if role == "user" and (is_summary_message(compacted) or is_synthetic_user_message(compacted)):
            continue
        compacted = compact_message_text(compacted)
        if not compacted:
            continue
        trimmed.append({"role": str(role), "content": compacted, "source": str(msg.get("source") or "")})

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
            "provider timeout",
            "provider error:",
        )
    )


def is_summary_message(text: str) -> bool:
    """Return True when the message is a compaction summary."""

    summary_text = text.strip()
    return summary_text.startswith(SUMMARY_PREFIX)


def is_synthetic_user_message(text: str) -> bool:
    """Detect injected provider bootstrap payloads that are not user intent.

    Older sessions may still contain composed first-turn bootstrap prompts in user
    history. We filter these out during compaction so retained history represents
    what the human asked, not provider harness scaffolding.
    """

    lowered = text.strip().lower()
    if not lowered:
        return False
    if lowered.startswith("<environment_context>") or lowered.startswith("<turn_aborted>"):
        return True
    return any(hint in lowered for hint in _SYNTHETIC_BOOTSTRAP_HINTS)


def collect_user_messages(history: list[ChatMessage]) -> list[str]:
    """Collect non-summary user messages for compacted history rebuild."""

    messages: list[str] = []
    for msg in history:
        if msg.get("role") != "user":
            continue
        if bool(msg.get("synthetic")):
            continue
        content = str(msg.get("content") or "").strip()
        if not content or is_summary_message(content) or is_synthetic_user_message(content):
            continue
        messages.append(content)
    return messages


def build_compacted_history(
    user_messages: list[str],
    summary_text: str,
    max_tokens: int,
    *,
    checkpoint: CompactionCheckpoint | None = None,
) -> list[ChatMessage]:
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

    history: list[ChatMessage] = [
        {"role": "user", "content": message, "source": "user"} for message in selected if message.strip()
    ]

    summary = summary_text.strip() if summary_text else ""
    if not summary:
        summary = "(no summary available)"
    if not summary.startswith(SUMMARY_PREFIX):
        summary = f"{SUMMARY_PREFIX}\n{summary}"
    summary_msg: ChatMessage = {
        "role": "user",
        "content": summary,
        "source": "compaction_summary",
        "synthetic": True,
    }
    if checkpoint is not None:
        summary_msg["checkpoint"] = checkpoint.model_dump()
    history.append(summary_msg)
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


def _extract_json_candidate(text: str) -> str | None:
    """Extract a JSON object candidate from model text output."""

    stripped = text.strip()
    if stripped.startswith("```"):
        fenced = stripped.split("```")
        for part in fenced:
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{") and candidate.endswith("}"):
                return candidate
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    return match.group(0).strip() if match else None


def _coerce_list(values: Any, limit: int = 8) -> list[str]:
    """Normalize arbitrary JSON fields into a bounded list of short strings."""

    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return []

    normalized: list[str] = []
    for item in values:
        text = str(item).strip()
        if not text:
            continue
        if text not in normalized:
            normalized.append(text[:280])
        if len(normalized) >= limit:
            break
    return normalized


def _checkpoint_from_raw_text(summary: str) -> CompactionCheckpoint | None:
    """Parse model summary text into a structured checkpoint if possible."""

    summary_content = (summary or "").strip()
    if summary_rejected(summary_content):
        return None

    candidate = _extract_json_candidate(summary_content)
    if candidate:
        try:
            payload = json.loads(candidate)
            if isinstance(payload, dict):
                checkpoint_payload = {
                    "progress": _coerce_list(payload.get("progress")),
                    "key_decisions": _coerce_list(payload.get("key_decisions") or payload.get("decisions")),
                    "user_preferences": _coerce_list(payload.get("user_preferences") or payload.get("preferences")),
                    "remaining_steps": _coerce_list(payload.get("remaining_steps") or payload.get("next_steps")),
                    "critical_artifacts": _coerce_list(payload.get("critical_artifacts") or payload.get("artifacts")),
                    "risks": _coerce_list(payload.get("risks")),
                }
                return CompactionCheckpoint.model_validate(checkpoint_payload)
        except (json.JSONDecodeError, ValidationError):
            pass

    # Keep non-empty plain text summaries by mapping them into `progress`; this
    # avoids throwing away useful model output just because JSON formatting drifted.
    if summary_content:
        return CompactionCheckpoint(progress=[summary_content[:600]])
    return None


def _build_fallback_checkpoint(history: list[ChatMessage]) -> CompactionCheckpoint:
    """Build a structured fallback checkpoint from existing history."""

    progress: list[str] = []
    key_decisions: list[str] = []
    user_preferences: list[str] = []
    remaining_steps: list[str] = []
    critical_artifacts: list[str] = []

    last_user = ""
    last_assistant = ""
    for msg in history:
        role = str(msg.get("role") or "")
        content = str(msg.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            last_user = content
        elif role == "assistant":
            last_assistant = content

        if content.startswith("Updated file "):
            critical_artifacts.append(content.splitlines()[0][:220])
        elif content.startswith("Ran command:"):
            critical_artifacts.append(content.splitlines()[0][:220])
        elif content.startswith("Delegated to "):
            key_decisions.append(content.splitlines()[0][:220])
        elif "should" in content.lower() and role == "assistant":
            key_decisions.append(content.splitlines()[0][:220])

    if last_user:
        remaining_steps.append(f"Primary user request: {last_user[:220]}")
    if last_assistant:
        progress.append(f"Latest assistant state: {last_assistant[:220]}")
    if not progress and critical_artifacts:
        progress.append("Applied changes/commands are listed under critical artifacts.")
    if not (progress or key_decisions or remaining_steps or critical_artifacts):
        progress.append("Summary unavailable; conversation was compacted due to context limits.")

    return CompactionCheckpoint(
        progress=_coerce_list(progress),
        key_decisions=_coerce_list(key_decisions),
        user_preferences=_coerce_list(user_preferences),
        remaining_steps=_coerce_list(remaining_steps),
        critical_artifacts=_coerce_list(critical_artifacts),
        risks=[],
    )


def _checkpoint_to_summary_text(checkpoint: CompactionCheckpoint, *, fallback_generated: bool) -> str:
    """Render a compact, model-friendly summary block from a checkpoint."""

    title = "Auto summary:" if fallback_generated else "Compaction checkpoint:"
    sections = [
        ("Progress", checkpoint.progress),
        ("Key decisions", checkpoint.key_decisions),
        ("User preferences", checkpoint.user_preferences),
        ("Remaining steps", checkpoint.remaining_steps),
        ("Critical artifacts", checkpoint.critical_artifacts),
        ("Risks", checkpoint.risks),
    ]

    lines = [title]
    for header, values in sections:
        if not values:
            continue
        lines.append(f"{header}:")
        for value in values:
            lines.append(f"- {value}")
    if len(lines) == 1:
        lines.append("- (no summary available)")
    return "\n".join(lines)


def _is_context_overflow_response(response_text: str) -> bool:
    """Detect provider errors caused by context-window overflow.

    `stream_with_runner` returns provider failures as text, so compaction must
    classify these failures and retry with a smaller history slice.
    """

    lowered = (response_text or "").strip().lower()
    if not lowered.startswith("provider error:"):
        return False
    return any(hint in lowered for hint in _CONTEXT_OVERFLOW_ERROR_HINTS)


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
        usage_since_compaction = state.usage_total_tokens_since_compaction
        estimated = max(estimated, usage_total or 0, usage_since_compaction)
        if estimated <= token_limit:
            return
        with log_ctx(session_id=session_id, model_id=model_id):
            log_event(
                logger,
                "prompt.history.compact",
                level=logging.DEBUG,
                estimated_tokens=estimated,
                usage_total_tokens=state.last_usage_total_tokens,
                usage_since_compaction_tokens=usage_since_compaction,
                token_limit=token_limit,
                context_limit=context_limit,
            )

    if not state.history:
        return

    async def _noop(_: str) -> None:
        return None

    compaction_history = prepare_compaction_history(state.history, context_limit)
    if not compaction_history:
        return

    summary_text = ""
    overflow_retries = 0
    # Retry compaction when providers reject overlong context windows.
    # We trim oldest entries one-by-one to preserve recency and keep retry
    # behavior deterministic for debugging and tests.
    while compaction_history and overflow_retries <= _COMPACTION_MAX_OVERFLOW_RETRIES:
        summary_text, _ = await stream_with_runner(
            runner,
            COMPACT_PROMPT,
            _noop,
            _noop,
            asyncio.Event(),
            history=compaction_history,
            log_context="history_compact",
        )
        summary_candidate = (summary_text or "").strip()
        if not _is_context_overflow_response(summary_candidate):
            break
        overflow_retries += 1
        removed = compaction_history.pop(0)
        with log_ctx(session_id=session_id, model_id=model_id):
            log_event(
                logger,
                "prompt.history.compact.retry",
                level=logging.WARNING,
                retry=overflow_retries,
                removed_role=str(removed.get("role") or ""),
                removed_preview=str(removed.get("content") or "")[:120].replace("\n", "\\n"),
                remaining_messages=len(compaction_history),
            )

    checkpoint = _checkpoint_from_raw_text(summary_text)
    fallback_generated = False
    if checkpoint is None:
        checkpoint = _build_fallback_checkpoint(state.history)
        fallback_generated = True
        with log_ctx(session_id=session_id, model_id=model_id):
            log_event(logger, "prompt.history.compact.fallback", level=logging.DEBUG)

    summary_content = _checkpoint_to_summary_text(checkpoint, fallback_generated=fallback_generated)
    user_messages = collect_user_messages(state.history)
    old_history_len = len(state.history)
    state.history = build_compacted_history(
        user_messages,
        summary_content,
        compact_user_message_max_tokens,
        checkpoint=checkpoint,
    )
    state.last_compaction_checkpoint = checkpoint.model_dump()
    # Reset the usage growth signal to the compacted history size so future
    # compaction decisions reflect post-compaction transcript pressure.
    state.usage_total_tokens_since_compaction = estimate_history_tokens(state.history)

    with log_ctx(session_id=session_id, model_id=model_id):
        log_event(
            logger,
            "prompt.history.compact.complete",
            level=logging.DEBUG,
            old_history_messages=old_history_len,
            new_history_messages=len(state.history),
            overflow_retries=overflow_retries,
            fallback_generated=fallback_generated,
        )

    if session_id:
        await env.send_notification(
            session_id,
            "Context compacted to stay within model limits. Recent details may be summarized.",
        )
