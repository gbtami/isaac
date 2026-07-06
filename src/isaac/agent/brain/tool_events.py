"""Helpers for tool call classification and history summaries."""

from __future__ import annotations

import contextlib
import json
from typing import Any

from pydantic import BaseModel


def is_delegate_tool(tool_name: str) -> bool:
    """Check if a tool name belongs to a registered delegate tool."""

    if not tool_name:
        return False
    try:
        from isaac.agent.subagents import DELEGATE_TOOL_HANDLERS
    except Exception:
        return False
    return tool_name in DELEGATE_TOOL_HANDLERS


def tool_history_summary(
    tool_name: str, raw_output: dict[str, Any], status: str, raw_input: dict[str, Any] | None = None
) -> str | None:
    """Build a compact history summary for a tool call."""

    content = raw_output.get("content")
    summary_prefix = f"{tool_name} ({status})"

    def _truncate(text: object, limit: int = 1200) -> str:
        text_str = str(text).strip()
        if len(text_str) <= limit:
            return text_str
        return text_str[:limit].rstrip() + "..."

    def _lines(*parts: str) -> str:
        return "\n".join(part for part in parts if part)

    if tool_name == "run_command":
        cmd = raw_output.get("command") or raw_output.get("cmd") or (raw_input or {}).get("command")
        cwd = raw_output.get("cwd") or (raw_input or {}).get("cwd")
        cmd_str = str(cmd).strip() if cmd else ""
        cwd_str = f" (cwd: {cwd})" if cwd else ""
        header = f"Ran command: {cmd_str}{cwd_str} [{status}]" if cmd_str else f"Ran command [{status}]"
        stdout = raw_output.get("content") or ""
        stderr = raw_output.get("error") or ""
        output = _lines(
            header,
            f"Stdout:\n{_truncate(stdout)}" if stdout else "",
            f"Stderr:\n{_truncate(stderr)}" if stderr else "",
        )
        return output
    if tool_name in {"edit_file", "apply_patch"}:
        path = raw_output.get("path") or (raw_input or {}).get("path")
        path_str = f" {path}" if path else ""
        summary = raw_output.get("content") or ""
        diff = raw_output.get("diff") or ""
        detail = ""
        if diff:
            detail = f"Diff:\n{_truncate(diff, 1600)}"
        elif summary:
            detail = f"Summary:\n{_truncate(summary, 800)}"
        return _lines(f"Updated file{path_str} [{status}]", detail)
    if tool_name == "read_file":
        path = raw_output.get("path") or (raw_input or {}).get("path")
        line_hint = ""
        start_line = raw_output.get("start_line")
        end_line = raw_output.get("end_line")
        total_lines = raw_output.get("total_lines")
        if isinstance(start_line, int) and isinstance(end_line, int) and end_line >= start_line:
            total_hint = f" of {total_lines}" if isinstance(total_lines, int) and total_lines >= end_line else ""
            line_hint = f" lines {start_line}-{end_line}{total_hint}"
        elif raw_input:
            start = raw_input.get("start")
            lines = raw_input.get("lines")
            if start is not None and lines is not None:
                try:
                    line_hint = f" lines {start}-{int(start) + int(lines) - 1}"
                except (TypeError, ValueError):
                    line_hint = f" from line {start}"
            elif start is not None:
                line_hint = f" from line {start}"
        sha = raw_output.get("sha256")
        sha_line = f"SHA256: {sha}" if sha else ""
        continuation = ""
        if raw_output.get("truncated"):
            next_start = raw_output.get("next_start")
            continuation = f"Truncated; continue with start={next_start}." if next_start else "Truncated."
        excerpt = _truncate(raw_output.get("content") or "", 600)
        detail = _lines(sha_line, continuation, f"Excerpt:\n{excerpt}" if excerpt else "")
        if path:
            return _lines(f"Read file {path}{line_hint} [{status}]", detail)
    if tool_name == "list_files":
        root = raw_output.get("directory") or raw_output.get("path") or (raw_input or {}).get("directory")
        listing = _truncate(raw_output.get("content") or "", 900)
        detail = f"Entries:\n{listing}" if listing else ""
        if root:
            return _lines(f"Listed files in {root} [{status}]", detail)
    if tool_name == "file_summary":
        path = raw_output.get("path") or (raw_input or {}).get("path")
        summary = _truncate(raw_output.get("content") or "", 1000)
        detail = f"Summary:\n{summary}" if summary else ""
        if path:
            return _lines(f"Summarized file {path} [{status}]", detail)
    if is_delegate_tool(tool_name):
        task = raw_output.get("task") or (raw_input or {}).get("task")
        task_str = f": {task}" if task else ""
        payload = _delegate_payload(raw_output)
        summary = raw_output.get("summary") or payload.get("summary") or raw_output.get("content") or ""
        artifact_detail = _delegate_artifact_detail(payload)
        detail = _lines(
            f"Summary:\n{_truncate(summary)}" if summary else "",
            artifact_detail,
        )
        return _lines(f"Delegated to {tool_name}{task_str} [{status}]", detail)
    if tool_name == "code_search":
        pattern = raw_output.get("pattern") or (raw_input or {}).get("pattern")
        directory = raw_output.get("directory") or (raw_input or {}).get("directory")
        matches = _truncate(raw_output.get("content") or "", 1000)
        match_count = raw_output.get("match_count")
        shown_count = raw_output.get("shown_count")
        count_detail = ""
        if isinstance(match_count, int):
            if isinstance(shown_count, int) and shown_count != match_count:
                count_detail = f"Showing {shown_count} of {match_count} matches."
            else:
                count_detail = f"Found {match_count} matches."
        if pattern:
            where = f" in {directory}" if directory else ""
            detail = _lines(count_detail, f"Matches:\n{matches}" if matches else "")
            return _lines(f"Searched for '{pattern}'{where} [{status}]", detail)
    if tool_name == "mark_plan_step":
        step = raw_output.get("step") or raw_output.get("step_id") or (raw_input or {}).get("step")
        step_status = raw_output.get("status") or (raw_input or {}).get("status")
        note = raw_output.get("note") or (raw_input or {}).get("note")
        detail = f" Note: {_truncate(note, 400)}" if note else ""
        if step and step_status:
            return f"Plan step {step} marked {step_status} [{status}].{detail}"
        return f"Plan progress reported [{status}].{detail}"
    if tool_name == "fetch_url":
        fetched = raw_output.get("url") or raw_output.get("source") or raw_output.get("request_url")
        status_code = raw_output.get("status_code")
        status_detail = f" ({status_code})" if status_code else ""
        content_detail = _truncate(raw_output.get("content") or "", 900)
        detail = f"Content:\n{content_detail}" if content_detail else ""
        if fetched:
            return _lines(f"Fetched URL {fetched}{status_detail} [{status}]", detail)
    if tool_name:
        return f"{summary_prefix}"
    if content:
        return f"Tool result [{status}]: {content}"
    return None


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


def _delegate_artifact_detail(payload: dict[str, Any]) -> str:
    if not payload:
        return ""
    counts: list[str] = []
    for key, label in (
        ("files", "files"),
        ("findings", "findings"),
        ("tests", "tests"),
        ("risks", "risks"),
        ("followups", "followups"),
        ("entries", "plan entries"),
    ):
        value = payload.get(key)
        if isinstance(value, list) and value:
            counts.append(f"{label}={len(value)}")
    if not counts:
        return ""
    return f"Artifacts: {', '.join(counts)}"


def should_record_tool_history(tool_name: str) -> bool:
    """Return True when a compact tool observation should enter brain history."""

    if is_delegate_tool(tool_name):
        return True
    return tool_name in {
        "edit_file",
        "apply_patch",
        "run_command",
        "read_file",
        "list_files",
        "file_summary",
        "code_search",
        "fetch_url",
        "mark_plan_step",
    }


def tool_kind(tool_name: str) -> str:
    """Return a tool kind label for a tool name."""

    name = tool_name.lower().strip()
    if name in {"read_file", "list_files", "file_summary"}:
        return "read"
    if name in {"edit_file", "apply_patch"}:
        return "edit"
    if name in {"code_search"}:
        return "search"
    if name in {"run_command"}:
        return "execute"
    if name in {"fetch_url"}:
        return "fetch"
    if name in {"mark_plan_step"}:
        return "plan"
    if is_delegate_tool(name):
        return "think"
    return "other"
