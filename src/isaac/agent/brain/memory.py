"""Structured coding memory for long-running sessions.

The chat transcript remains the canonical conversation, but coding agents also
need durable task facts that are easier to select than prose-only history:
files read, files edited, commands run, searches performed, delegate outcomes,
and similar observations.  This module keeps those facts as typed Pydantic
models and renders a small, relevant per-run context block for Pydantic AI.
"""

from __future__ import annotations

import contextlib
from datetime import datetime, timezone
import json
import re
from typing import Any, ClassVar, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from isaac.agent.brain.tool_events import is_delegate_tool, tool_history_summary, tool_kind

MemoryKind = Literal[
    "observation",
    "edit",
    "command",
    "delegate",
    "plan",
    "fetch",
    "test",
    "risk",
    "finding",
    "followup",
    "other",
]

_MAX_EVENT_SUMMARY_TOKENS = 420
_DEFAULT_MEMORY_BUDGET_TOKENS = 3_000
_MAX_MEMORY_BUDGET_TOKENS = 6_000
_MAX_SELECTED_EVENTS = 36
_DEFAULT_MAX_EVENTS = 600
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


def _approx_text_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def _truncate_text_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0 or _approx_text_tokens(text) <= max_tokens:
        return text
    limit = max(1, max_tokens * 4)
    return text[:limit].rstrip() + "…"


class CodingMemoryEvent(BaseModel):
    """A compact, durable coding fact discovered during a session."""

    model_config = ConfigDict(extra="ignore")

    event_id: str = Field(default_factory=lambda: uuid4().hex)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    kind: MemoryKind
    summary: str
    status: str = "completed"
    tool_name: str | None = None
    tool_kind: str | None = None
    paths: list[str] = Field(default_factory=list)
    command: str | None = None
    query: str | None = None
    url: str | None = None
    sha256: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    IMPORTANT_KINDS: ClassVar[set[str]] = {
        "edit",
        "command",
        "delegate",
        "plan",
        "test",
        "risk",
        "finding",
        "followup",
    }

    @property
    def is_important(self) -> bool:
        return self.kind in self.IMPORTANT_KINDS or self.status != "completed"

    def compact_summary(self, *, max_tokens: int = _MAX_EVENT_SUMMARY_TOKENS) -> str:
        return _truncate_text_tokens(self.summary.strip(), max_tokens)

    def search_text(self) -> str:
        parts = [self.kind, self.status, self.tool_name or "", self.summary]
        parts.extend(self.paths)
        parts.extend(str(v) for v in (self.command, self.query, self.url, self.sha256) if v)
        return "\n".join(parts).lower()


class CodingMemory(BaseModel):
    """Bounded collection of coding memory events."""

    model_config = ConfigDict(extra="ignore")

    events: list[CodingMemoryEvent] = Field(default_factory=list)
    max_events: int = _DEFAULT_MAX_EVENTS

    def append(self, event: CodingMemoryEvent) -> None:
        self.events.append(event)
        self.trim()

    def extend(self, events: list[CodingMemoryEvent]) -> None:
        if not events:
            return
        self.events.extend(events)
        self.trim()

    def trim(self) -> None:
        if self.max_events <= 0:
            self.events = []
            return
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events :]

    def model_dump_jsonable(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_snapshot(cls, data: Any) -> CodingMemory:
        if isinstance(data, CodingMemory):
            return data
        if not isinstance(data, dict):
            return cls()
        try:
            return cls.model_validate(data)
        except Exception:
            return cls()


class TaskFileSummary(BaseModel):
    """Compact per-file task state derived from coding memory."""

    model_config = ConfigDict(extra="ignore")

    path: str
    status: str = "observed"
    notes: list[str] = Field(default_factory=list)
    last_sha256: str | None = None


class TaskCommandSummary(BaseModel):
    """Compact command/test state derived from coding memory."""

    model_config = ConfigDict(extra="ignore")

    command: str
    status: str = "completed"
    summary: str


class TaskCheckpoint(BaseModel):
    """Deterministic task checkpoint used as compact run context.

    This is intentionally derived from structured events instead of model prose.
    It gives the next model run a stable map of important files, validation,
    and unresolved work without waiting for emergency context compaction.
    """

    model_config = ConfigDict(extra="ignore")

    key_files: list[TaskFileSummary] = Field(default_factory=list)
    recent_commands: list[TaskCommandSummary] = Field(default_factory=list)
    open_items: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    recent_progress: list[str] = Field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (self.key_files or self.recent_commands or self.open_items or self.risks or self.recent_progress)


def memory_events_from_tool_result(
    tool_name: str,
    raw_output: dict[str, Any],
    status: str,
    *,
    raw_input: dict[str, Any] | None = None,
) -> list[CodingMemoryEvent]:
    """Extract structured coding-memory events from a tool result."""

    if not tool_name:
        return []
    kind = _memory_kind_for_tool(tool_name)
    if kind is None:
        return []

    summary = tool_history_summary(tool_name, raw_output, status, raw_input=raw_input)
    if not summary:
        return []
    summary = _truncate_text_tokens(summary, _MAX_EVENT_SUMMARY_TOKENS)

    event = CodingMemoryEvent(
        kind=kind,
        summary=summary,
        status=status,
        tool_name=tool_name,
        tool_kind=tool_kind(tool_name),
        paths=_extract_paths(tool_name, raw_output, raw_input),
        command=_extract_command(tool_name, raw_output, raw_input),
        query=_extract_query(tool_name, raw_output, raw_input),
        url=_extract_url(tool_name, raw_output, raw_input),
        sha256=_extract_sha256(raw_output),
        metadata=_extract_metadata(tool_name, raw_output, raw_input),
    )
    if is_delegate_tool(tool_name):
        return [event, *_delegate_artifact_events(tool_name, raw_output, status, raw_input)]
    return [event]


def delegate_artifact_paths(raw_output: dict[str, Any]) -> list[str]:
    """Return file paths declared by a delegate's structured artifact payload."""

    payload = _delegate_payload(raw_output)
    paths: list[str] = []
    for file_item in _dict_items(payload.get("files")):
        path = _str_or_none(file_item.get("path") or file_item.get("file"))
        if path:
            paths.append(path)
    for finding in _dict_items(payload.get("findings")):
        path = _str_or_none(finding.get("file") or finding.get("path"))
        if path:
            paths.append(path)
    return _unique(paths)


def build_task_checkpoint(
    memory: CodingMemory,
    *,
    current_prompt: str = "",
    max_files: int = 8,
    max_commands: int = 5,
    max_open_items: int = 8,
    max_risks: int = 5,
    max_progress: int = 6,
) -> TaskCheckpoint:
    """Build a deterministic task checkpoint from structured memory events.

    The selector intentionally favors unresolved work, failed validation, recent
    edits, and prompt-relevant paths.  Unlike compaction, this does not call a
    model; it is safe to run for every prompt and survives restart because the
    source events are persisted in ``CodingMemory``.
    """

    events = list(memory.events)
    if not events:
        return TaskCheckpoint()

    prompt_terms = _terms(current_prompt)
    key_files = _checkpoint_files(events, prompt_terms=prompt_terms, max_files=max_files)
    recent_commands = _checkpoint_commands(events, max_commands=max_commands)
    open_items = _checkpoint_open_items(events, max_items=max_open_items)
    risks = _checkpoint_risks(events, max_items=max_risks)
    recent_progress = _checkpoint_recent_progress(events, max_items=max_progress)
    return TaskCheckpoint(
        key_files=key_files,
        recent_commands=recent_commands,
        open_items=open_items,
        risks=risks,
        recent_progress=recent_progress,
    )


def task_checkpoint_context(
    memory: CodingMemory,
    *,
    current_prompt: str,
    context_limit: int | None,
) -> str | None:
    """Render the deterministic task checkpoint as transient run context."""

    checkpoint = build_task_checkpoint(memory, current_prompt=current_prompt)
    if checkpoint.is_empty:
        return None

    max_tokens = _checkpoint_budget(context_limit)
    lines = [
        "Session task checkpoint (deterministic summary from structured coding memory):",
        "Use this for continuity; verify file contents before exact edits.",
    ]

    if checkpoint.open_items:
        lines.append("Open work:")
        lines.extend(f"- {item}" for item in checkpoint.open_items)
    if checkpoint.risks:
        lines.append("Risks / unresolved concerns:")
        lines.extend(f"- {risk}" for risk in checkpoint.risks)
    if checkpoint.key_files:
        lines.append("Key files:")
        for file_summary in checkpoint.key_files:
            note = "; ".join(file_summary.notes[:2])
            sha = f" sha256={file_summary.last_sha256[:12]}" if file_summary.last_sha256 else ""
            suffix = f" — {note}" if note else ""
            lines.append(f"- {file_summary.path} [{file_summary.status}{sha}]{suffix}")
    if checkpoint.recent_commands:
        lines.append("Recent validation / commands:")
        for command_summary in checkpoint.recent_commands:
            summary = command_summary.summary.replace("\n", " ")
            lines.append(f"- {command_summary.command} [{command_summary.status}]: {summary}")
    if checkpoint.recent_progress:
        lines.append("Recent progress:")
        lines.extend(f"- {item}" for item in checkpoint.recent_progress)

    rendered = "\n".join(lines)
    return _truncate_text_tokens(rendered, max_tokens)


def selected_memory_context(
    memory: CodingMemory,
    *,
    current_prompt: str,
    context_limit: int | None,
    max_events: int = _MAX_SELECTED_EVENTS,
) -> str | None:
    """Render relevant structured memory as transient Pydantic AI context."""

    events = select_memory_events(
        memory.events,
        current_prompt=current_prompt,
        context_limit=context_limit,
        max_events=max_events,
    )
    if not events:
        return None

    lines = [
        "Session coding memory (durable structured facts from earlier tool use):",
        "Use these as background for continuity. Re-read files before exact edits if contents may have changed.",
    ]
    for event in events:
        label = event.kind
        if event.status != "completed":
            label = f"{label}/{event.status}"
        locator = _event_locator(event)
        prefix = f"- [{label}] {locator}:" if locator else f"- [{label}]:"
        text = event.compact_summary(max_tokens=180).replace("\n", " ")
        lines.append(f"{prefix} {text}".rstrip())
    return "\n".join(lines)


def select_memory_events(
    events: list[CodingMemoryEvent],
    *,
    current_prompt: str,
    context_limit: int | None,
    max_events: int = _MAX_SELECTED_EVENTS,
) -> list[CodingMemoryEvent]:
    """Select memory events by importance, relevance, recency, and budget."""

    if not events or max_events <= 0:
        return []

    prompt_terms = _terms(current_prompt)
    last_index = len(events) - 1
    scored: list[tuple[int, int, CodingMemoryEvent]] = []
    recent_start = max(0, len(events) - 20)
    for idx, event in enumerate(events):
        score = _event_score(event, idx=idx, last_index=last_index, recent_start=recent_start, prompt_terms=prompt_terms)
        if score <= 0:
            continue
        scored.append((score, idx, event))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    budget = _memory_budget(context_limit)
    selected: list[tuple[int, CodingMemoryEvent]] = []
    used_tokens = 0
    for _score, idx, event in scored:
        if len(selected) >= max_events:
            break
        tokens = _approx_text_tokens(event.compact_summary()) + 16
        if selected and used_tokens + tokens > budget:
            continue
        selected.append((idx, event))
        used_tokens += tokens

    return [event for _idx, event in sorted(selected, key=lambda item: item[0])]


def _memory_kind_for_tool(tool_name: str) -> MemoryKind | None:
    name = tool_name.lower().strip()
    if name in {"edit_file", "apply_patch"}:
        return "edit"
    if name == "run_command":
        return "command"
    if name in {"read_file", "list_files", "file_summary", "code_search"}:
        return "observation"
    if name == "fetch_url":
        return "fetch"
    if name in {"planner", "mark_plan_step"}:
        return "plan"
    if is_delegate_tool(name):
        return "delegate"
    return None


def _extract_paths(tool_name: str, raw_output: dict[str, Any], raw_input: dict[str, Any] | None) -> list[str]:
    paths: list[str] = []
    for key in ("path", "file", "target_path", "directory"):
        value = raw_output.get(key) or (raw_input or {}).get(key)
        if isinstance(value, str) and value.strip():
            paths.append(value.strip())
    if is_delegate_tool(tool_name):
        paths.extend(delegate_artifact_paths(raw_output))
    if tool_name == "apply_patch":
        # apply_patch results may only have diff text; keep path extraction conservative.
        diff = str(raw_output.get("diff") or raw_output.get("content") or "")
        for match in re.finditer(r"^diff --git a/(\S+) b/\S+|^\+\+\+ b/(\S+)", diff, re.MULTILINE):
            path = next((group for group in match.groups() if group), "")
            if path:
                paths.append(path)
    return _unique(paths)


def _extract_command(tool_name: str, raw_output: dict[str, Any], raw_input: dict[str, Any] | None) -> str | None:
    if tool_name != "run_command":
        return None
    value = raw_output.get("command") or raw_output.get("cmd") or (raw_input or {}).get("command")
    return str(value).strip() if value else None


def _extract_query(tool_name: str, raw_output: dict[str, Any], raw_input: dict[str, Any] | None) -> str | None:
    if tool_name == "code_search":
        value = raw_output.get("pattern") or (raw_input or {}).get("pattern")
    elif tool_name == "list_files":
        value = raw_output.get("directory") or (raw_input or {}).get("directory")
    else:
        value = None
    return str(value).strip() if value else None


def _extract_url(tool_name: str, raw_output: dict[str, Any], raw_input: dict[str, Any] | None) -> str | None:
    if tool_name != "fetch_url":
        return None
    value = raw_output.get("url") or raw_output.get("source") or raw_output.get("request_url") or (raw_input or {}).get("url")
    return str(value).strip() if value else None


def _extract_sha256(raw_output: dict[str, Any]) -> str | None:
    value = raw_output.get("sha256")
    return str(value).strip() if value else None


def _extract_metadata(tool_name: str, raw_output: dict[str, Any], raw_input: dict[str, Any] | None) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if raw_input:
        for key in ("start", "lines", "max_lines", "cwd", "expected_sha256", "max_results"):
            if key in raw_input and raw_input[key] is not None:
                metadata[key] = raw_input[key]
    for key in (
        "returncode",
        "truncated",
        "num_tokens",
        "status_code",
        "total_lines",
        "start_line",
        "end_line",
        "next_start",
        "match_count",
        "shown_count",
        "max_results",
    ):
        if key in raw_output and raw_output[key] is not None:
            metadata[key] = raw_output[key]
    if tool_name == "run_command" and raw_output.get("error"):
        metadata["error"] = str(raw_output.get("error"))[:500]
    if is_delegate_tool(tool_name):
        metadata["delegate_tool"] = tool_name
        for key in ("delegate_session_id", "delegate_run_id"):
            value = raw_output.get(key)
            if value:
                metadata[key] = str(value)
    return metadata


def _delegate_artifact_events(
    tool_name: str,
    raw_output: dict[str, Any],
    status: str,
    raw_input: dict[str, Any] | None,
) -> list[CodingMemoryEvent]:
    """Expand structured delegate output into parent-session memory events.

    Delegate agents already return a compact summary, but their structured JSON
    contains more durable coding facts than a prose blob can preserve: file
    artifacts, review findings, tests, risks, follow-ups, and planner entries.
    These events make the parent agent's future context depend on typed facts
    rather than only the delegate's summary paragraph.
    """

    payload = _delegate_payload(raw_output)
    if not payload:
        return []

    events: list[CodingMemoryEvent] = []
    base_metadata = _delegate_base_metadata(tool_name, raw_output, raw_input)

    for index, file_item in enumerate(_dict_items(payload.get("files"))):
        path = _str_or_none(file_item.get("path") or file_item.get("file"))
        summary = _str_or_none(file_item.get("summary") or file_item.get("description"))
        intent = _str_or_none(file_item.get("intent") or file_item.get("rationale"))
        action = _str_or_none(file_item.get("action") or file_item.get("status"))
        if not path and not summary:
            continue
        kind: MemoryKind = "edit" if _delegate_file_action_is_edit(tool_name, action, summary) else "observation"
        text = _join_sentences(
            f"Delegate {tool_name} reported file artifact {path or '(unknown file)'}.",
            f"Action: {action}." if action else "",
            f"Summary: {summary}." if summary else "",
            f"Intent: {intent}." if intent else "",
        )
        events.append(
            CodingMemoryEvent(
                kind=kind,
                summary=_truncate_text_tokens(text, _MAX_EVENT_SUMMARY_TOKENS),
                status=status,
                tool_name=tool_name,
                tool_kind=tool_kind(tool_name),
                paths=[path] if path else [],
                metadata={**base_metadata, "artifact_type": "file", "artifact_index": index, **_maybe("action", action)},
            )
        )

    for index, finding in enumerate(_dict_items(payload.get("findings"))):
        description = _str_or_none(finding.get("description") or finding.get("summary") or finding.get("issue"))
        if not description:
            continue
        path = _str_or_none(finding.get("file") or finding.get("path"))
        severity = _str_or_none(finding.get("severity"))
        line = finding.get("line")
        suggestion = _str_or_none(finding.get("suggestion") or finding.get("fix"))
        location = _format_location(path, line)
        text = _join_sentences(
            f"Delegate {tool_name} finding{f' at {location}' if location else ''}.",
            f"Severity: {severity}." if severity else "",
            description,
            f"Suggestion: {suggestion}." if suggestion else "",
        )
        events.append(
            CodingMemoryEvent(
                kind="finding",
                summary=_truncate_text_tokens(text, _MAX_EVENT_SUMMARY_TOKENS),
                status="open" if status == "completed" else status,
                tool_name=tool_name,
                tool_kind=tool_kind(tool_name),
                paths=[path] if path else [],
                metadata={
                    **base_metadata,
                    "artifact_type": "finding",
                    "artifact_index": index,
                    **_maybe("severity", severity),
                    **_maybe("line", line),
                },
            )
        )

    for index, test in enumerate(_string_items(payload.get("tests"))):
        events.append(
            CodingMemoryEvent(
                kind="test",
                summary=f"Delegate {tool_name} reported test/validation item: {test}",
                status="open" if tool_name == "review" and status == "completed" else status,
                tool_name=tool_name,
                tool_kind=tool_kind(tool_name),
                metadata={**base_metadata, "artifact_type": "test", "artifact_index": index},
            )
        )

    for index, risk in enumerate(_string_items(payload.get("risks"))):
        events.append(
            CodingMemoryEvent(
                kind="risk",
                summary=f"Delegate {tool_name} reported risk/open concern: {risk}",
                status="open" if status == "completed" else status,
                tool_name=tool_name,
                tool_kind=tool_kind(tool_name),
                metadata={**base_metadata, "artifact_type": "risk", "artifact_index": index},
            )
        )

    for index, followup in enumerate(_string_items(payload.get("followups") or payload.get("next_steps"))):
        events.append(
            CodingMemoryEvent(
                kind="followup",
                summary=f"Delegate {tool_name} recommended follow-up: {followup}",
                status="open" if status == "completed" else status,
                tool_name=tool_name,
                tool_kind=tool_kind(tool_name),
                metadata={**base_metadata, "artifact_type": "followup", "artifact_index": index},
            )
        )

    for index, entry in enumerate(_plan_entries(payload)):
        content = _str_or_none(entry.get("content") or entry.get("summary") or entry.get("task"))
        if not content:
            continue
        priority = _str_or_none(entry.get("priority"))
        events.append(
            CodingMemoryEvent(
                kind="plan",
                summary=_join_sentences(
                    f"Delegate {tool_name} planned step: {content}",
                    f"Priority: {priority}." if priority else "",
                ),
                status="open" if status == "completed" else status,
                tool_name=tool_name,
                tool_kind=tool_kind(tool_name),
                metadata={
                    **base_metadata,
                    "artifact_type": "plan_entry",
                    "artifact_index": index,
                    **_maybe("priority", priority),
                },
            )
        )

    return events


def _delegate_payload(raw_output: dict[str, Any]) -> dict[str, Any]:
    content = raw_output.get("content")
    if isinstance(content, BaseModel):
        with contextlib.suppress(Exception):
            return content.model_dump()
    if isinstance(content, dict):
        return content
    if isinstance(content, str):
        stripped = content.strip()
        if stripped.startswith("{"):
            with contextlib.suppress(Exception):
                parsed = json.loads(stripped)
                if isinstance(parsed, dict):
                    return parsed
    return {}


def _delegate_base_metadata(
    tool_name: str,
    raw_output: dict[str, Any],
    raw_input: dict[str, Any] | None,
) -> dict[str, Any]:
    metadata = _extract_metadata(tool_name, raw_output, raw_input)
    task = _str_or_none(raw_input.get("task") if raw_input else None)
    if task:
        metadata["delegate_task"] = _truncate_text_tokens(task, 120)
    return metadata


def _dict_items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    items: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, BaseModel):
            with contextlib.suppress(Exception):
                item = item.model_dump()
        if isinstance(item, dict):
            items.append(item)
    return items


def _string_items(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, BaseModel):
            with contextlib.suppress(Exception):
                item = item.model_dump()
            text = _str_or_none(item) or ""
        elif isinstance(item, dict):
            text = _str_or_none(item.get("summary") or item.get("content") or item.get("description")) or ""
        else:
            text = str(item).strip() if item is not None else ""
        if text:
            items.append(text)
    return items


def _plan_entries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    entries = _dict_items(payload.get("entries"))
    if entries:
        return entries
    plan = payload.get("plan")
    if isinstance(plan, dict):
        return _dict_items(plan.get("entries"))
    return []


def _delegate_file_action_is_edit(tool_name: str, action: str | None, summary: str | None) -> bool:
    action_text = (action or "").lower()
    summary_text = (summary or "").lower()
    edit_words = {"changed", "edited", "updated", "created", "deleted", "patched", "modified", "renamed"}
    read_words = {"read", "reviewed", "inspected", "checked", "summarized", "unchanged", "verified"}
    if action_text:
        if any(word in action_text for word in edit_words):
            return True
        if any(word in action_text for word in read_words):
            return False
    if any(word in summary_text for word in read_words) and not any(word in summary_text for word in edit_words):
        return False
    return tool_name == "coding"


def _format_location(path: str | None, line: Any) -> str:
    if path and line is not None:
        return f"{path}:{line}"
    if path:
        return path
    if line is not None:
        return f"line {line}"
    return ""


def _join_sentences(*parts: str) -> str:
    return " ".join(part.strip() for part in parts if part and part.strip())


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, dict):
        for key in ("summary", "content", "description", "text"):
            nested = _str_or_none(value.get(key))
            if nested:
                return nested
        return None
    text = str(value).strip()
    return text or None


def _maybe(key: str, value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, str) and not value.strip():
        return {}
    return {key: value}


def _checkpoint_files(
    events: list[CodingMemoryEvent],
    *,
    prompt_terms: set[str],
    max_files: int,
) -> list[TaskFileSummary]:
    if max_files <= 0:
        return []

    by_path: dict[str, tuple[int, int, TaskFileSummary]] = {}
    last_index = len(events) - 1
    for idx, event in enumerate(events):
        if not event.paths:
            continue
        for path in event.paths:
            status = _file_checkpoint_status(event)
            note = _checkpoint_note(event)
            score = _file_checkpoint_score(event, path, idx=idx, last_index=last_index, prompt_terms=prompt_terms)
            previous = by_path.get(path)
            if previous is None:
                by_path[path] = (score, idx, TaskFileSummary(path=path, status=status, notes=[note], last_sha256=event.sha256))
                continue
            old_score, old_idx, summary = previous
            if note not in summary.notes:
                summary.notes.insert(0, note)
                summary.notes = summary.notes[:3]
            if score >= old_score or idx >= old_idx:
                summary.status = status
                summary.last_sha256 = event.sha256 or summary.last_sha256
                by_path[path] = (max(score, old_score), idx, summary)

    ranked = sorted(by_path.values(), key=lambda item: (item[0], item[1]), reverse=True)
    return [summary for _score, _idx, summary in ranked[:max_files]]


def _checkpoint_commands(events: list[CodingMemoryEvent], *, max_commands: int) -> list[TaskCommandSummary]:
    summaries: list[TaskCommandSummary] = []
    for event in reversed(events):
        if event.kind not in {"command", "test"}:
            continue
        command = event.command or event.tool_name or "validation"
        summaries.append(
            TaskCommandSummary(
                command=command,
                status=event.status,
                summary=event.compact_summary(max_tokens=90).replace("\n", " "),
            )
        )
        if len(summaries) >= max_commands:
            break
    summaries.reverse()
    return summaries


def _checkpoint_open_items(events: list[CodingMemoryEvent], *, max_items: int) -> list[str]:
    items: list[str] = []
    for event in reversed(events):
        if event.kind not in {"finding", "followup", "plan", "test"}:
            continue
        if event.kind in {"plan", "test"} and event.status == "completed":
            continue
        text = _checkpoint_note(event)
        if not _append_unique(items, text):
            continue
        if len(items) >= max_items:
            break
    items.reverse()
    return items


def _checkpoint_risks(events: list[CodingMemoryEvent], *, max_items: int) -> list[str]:
    risks: list[str] = []
    for event in reversed(events):
        if event.kind != "risk" and not (event.status != "completed" and event.kind == "command"):
            continue
        text = _checkpoint_note(event)
        if not _append_unique(risks, text):
            continue
        if len(risks) >= max_items:
            break
    risks.reverse()
    return risks


def _checkpoint_recent_progress(events: list[CodingMemoryEvent], *, max_items: int) -> list[str]:
    progress: list[str] = []
    for event in reversed(events):
        if event.status != "completed":
            continue
        if event.kind not in {"edit", "delegate", "plan"}:
            continue
        text = _checkpoint_note(event)
        if not _append_unique(progress, text):
            continue
        if len(progress) >= max_items:
            break
    progress.reverse()
    return progress


def _file_checkpoint_status(event: CodingMemoryEvent) -> str:
    if event.status != "completed":
        return f"{event.kind}_{event.status}"
    if event.kind == "edit":
        return "edited"
    if event.kind in {"finding", "risk", "followup"}:
        return "needs_attention"
    if event.kind == "command":
        return "validated"
    return "observed"


def _file_checkpoint_score(
    event: CodingMemoryEvent,
    path: str,
    *,
    idx: int,
    last_index: int,
    prompt_terms: set[str],
) -> int:
    recency = max(0, min(25, last_index - idx))
    score = 25 - recency
    if event.kind == "edit":
        score += 80
    elif event.kind in {"finding", "risk", "followup"}:
        score += 72
    elif event.kind == "command":
        score += 42
    elif event.kind in {"observation", "fetch"}:
        score += 30
    else:
        score += 20
    if event.status != "completed":
        score += 24
    specific_terms = _specific_checkpoint_terms(prompt_terms)
    if _matches_terms(path.lower(), specific_terms) or _matches_terms(event.search_text(), specific_terms):
        score += 80
    return score


def _checkpoint_note(event: CodingMemoryEvent, *, max_tokens: int = 70) -> str:
    locator = _event_locator(event)
    prefix = f"{event.kind}"
    if event.status != "completed":
        prefix = f"{prefix}/{event.status}"
    if locator:
        prefix = f"{prefix} {locator}"
    body = event.compact_summary(max_tokens=max_tokens).replace("\n", " ")
    if body.lower().startswith(prefix.lower()):
        return body
    return f"{prefix}: {body}"


def _append_unique(items: list[str], text: str) -> bool:
    normalized = text.strip()
    if not normalized:
        return False
    key = re.sub(r"\s+", " ", normalized.lower())
    existing = {re.sub(r"\s+", " ", item.lower()) for item in items}
    if key in existing:
        return False
    items.append(normalized)
    return True


def _checkpoint_budget(context_limit: int | None) -> int:
    if not context_limit or context_limit <= 0:
        return 1_600
    return max(800, min(2_400, int(context_limit * 0.035)))


def _specific_checkpoint_terms(terms: set[str]) -> set[str]:
    """Drop generic directory words before scoring checkpoint relevance."""

    generic = {"src", "lib", "app", "tests", "test", "docs", "doc"}
    return {term for term in terms if len(term) >= 4 and term not in generic}


def _event_score(
    event: CodingMemoryEvent,
    *,
    idx: int,
    last_index: int,
    recent_start: int,
    prompt_terms: set[str],
) -> int:
    _ = last_index
    recent_bonus = 16 if idx >= recent_start else 0
    relevant = _matches_terms(event.search_text(), prompt_terms)
    score = recent_bonus
    if event.kind in {"edit", "command", "delegate", "plan", "test", "risk", "finding", "followup"}:
        score += 70
    elif event.kind in {"observation", "fetch"}:
        score += 44 if relevant else 12
    else:
        score += 16
    if relevant:
        score += 30
    if event.status != "completed":
        score += 22
    return score


def _memory_budget(context_limit: int | None) -> int:
    if not context_limit or context_limit <= 0:
        return _DEFAULT_MEMORY_BUDGET_TOKENS
    return max(1_200, min(_MAX_MEMORY_BUDGET_TOKENS, int(context_limit * 0.08)))


def _event_locator(event: CodingMemoryEvent) -> str:
    if event.paths:
        return ", ".join(event.paths[:3])
    if event.command:
        return event.command
    if event.query:
        return event.query
    if event.url:
        return event.url
    return event.tool_name or ""


def _terms(text: str) -> set[str]:
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
    return any(term in text for term in terms)


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        cleaned = value.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        unique.append(cleaned)
    return unique


__all__ = [
    "CodingMemory",
    "CodingMemoryEvent",
    "TaskCheckpoint",
    "TaskCommandSummary",
    "TaskFileSummary",
    "build_task_checkpoint",
    "delegate_artifact_paths",
    "memory_events_from_tool_result",
    "select_memory_events",
    "selected_memory_context",
    "task_checkpoint_context",
]
