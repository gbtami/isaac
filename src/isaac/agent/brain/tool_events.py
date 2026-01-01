"""Helpers for tool call classification and history summaries."""

from __future__ import annotations

from typing import Any


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
        if path:
            return f"Read file {path} [{status}]"
    if tool_name == "list_files":
        root = raw_output.get("directory") or raw_output.get("path") or (raw_input or {}).get("directory")
        if root:
            return f"Listed files in {root} [{status}]"
    if tool_name == "file_summary":
        path = raw_output.get("path") or (raw_input or {}).get("path")
        if path:
            return f"Summarized file {path} [{status}]"
    if is_delegate_tool(tool_name):
        task = raw_output.get("task") or (raw_input or {}).get("task")
        task_str = f": {task}" if task else ""
        summary = raw_output.get("summary") or raw_output.get("content") or ""
        detail = f"Summary:\n{_truncate(summary)}" if summary else ""
        return _lines(f"Delegated to {tool_name}{task_str} [{status}]", detail)
    if tool_name == "code_search":
        pattern = raw_output.get("pattern") or (raw_input or {}).get("pattern")
        directory = raw_output.get("directory") or (raw_input or {}).get("directory")
        if pattern:
            where = f" in {directory}" if directory else ""
            return f"Searched for '{pattern}'{where} [{status}]"
    if tool_name == "fetch_url":
        fetched = raw_output.get("url") or raw_output.get("source") or raw_output.get("request_url")
        status_code = raw_output.get("status_code")
        detail = f" ({status_code})" if status_code else ""
        if fetched:
            return f"Fetched URL {fetched}{detail} [{status}]"
    if tool_name:
        return f"{summary_prefix}"
    if content:
        return f"Tool result [{status}]: {content}"
    return None


def should_record_tool_history(tool_name: str) -> bool:
    """Keep history focused on state-changing actions and delegate outputs."""

    if is_delegate_tool(tool_name):
        return True
    return tool_name in {"edit_file", "apply_patch", "run_command"}


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
    if is_delegate_tool(name):
        return "think"
    return "other"
