"""Central tool safety and approval policy for Isaac tools.

The concrete filesystem/shell/network guards live in individual tool modules.
This module keeps the higher-level decision of *which* tool calls need user
approval in one place so Pydantic AI deferred approvals, direct ACP tool-call
blocks, and delegate sub-agent calls cannot drift apart.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlparse

ToolKind = Literal["read", "edit", "delete", "move", "search", "execute", "think", "fetch", "switch_mode", "other"]
ToolRisk = Literal["read", "write", "execute", "network", "delegate", "other"]

READ_ONLY_TOOL_NAMES = frozenset({"list_files", "read_file", "file_summary", "code_search"})
WRITE_TOOL_NAMES = frozenset({"edit_file", "apply_patch"})
EXECUTE_TOOL_NAMES = frozenset({"run_command"})
NETWORK_TOOL_NAMES = frozenset({"fetch_url"})
DELEGATE_TOOL_NAMES = frozenset({"planner", "coding", "review"})

_APPROVAL_ALWAYS_OPTION_BY_RISK: dict[ToolRisk, str] = {
    "write": "allow_this_target",
    "execute": "allow_this_command",
    "network": "allow_this_host",
    "delegate": "allow_this_delegate",
    "other": "allow_this_tool",
}


@dataclass(frozen=True)
class ToolPolicy:
    """Risk classification for one tool name."""

    risk: ToolRisk
    kind: ToolKind
    requires_approval: bool = False


def classify_tool(tool_name: str) -> ToolPolicy:
    """Return Isaac's approval-visible risk policy for a tool."""

    if tool_name in READ_ONLY_TOOL_NAMES:
        return ToolPolicy(risk="read", kind="search" if tool_name == "code_search" else "read")
    if tool_name in WRITE_TOOL_NAMES:
        return ToolPolicy(risk="write", kind="edit", requires_approval=True)
    if tool_name in EXECUTE_TOOL_NAMES:
        return ToolPolicy(risk="execute", kind="execute", requires_approval=True)
    if tool_name in NETWORK_TOOL_NAMES:
        return ToolPolicy(risk="network", kind="fetch", requires_approval=True)
    if tool_name in DELEGATE_TOOL_NAMES:
        # Delegates are visible agent-planning operations. Their inner risky
        # tools still go through the same approval policy, so approving the
        # delegate wrapper itself would mostly add noise.
        return ToolPolicy(risk="delegate", kind="think", requires_approval=False)
    return ToolPolicy(risk="other", kind="other", requires_approval=True)


def requires_tool_approval(tool_name: str, *, mode: str = "ask") -> bool:
    """Return true when a tool call should be gated in the current session mode."""

    if (mode or "ask").strip().lower() != "ask":
        return False
    return classify_tool(tool_name).requires_approval


def tool_permission_kind(tool_name: str) -> ToolKind:
    return classify_tool(tool_name).kind


def approval_always_option_id(tool_name: str) -> str:
    return _APPROVAL_ALWAYS_OPTION_BY_RISK.get(classify_tool(tool_name).risk, "allow_this_tool")


def _short(value: Any, limit: int = 160) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()}..."


def _stable_json(args: dict[str, Any]) -> str:
    try:
        return json.dumps(args, sort_keys=True, separators=(",", ":"), default=str)
    except TypeError:
        return str(sorted(args.items()))


def permission_cache_key(tool_name: str, args: dict[str, Any]) -> tuple[str, str]:
    """Return the allow-always cache key for a tool call.

    ``run_command`` deliberately keeps the historical key shape
    ``(command, cwd)`` so existing command approval caching semantics and tests
    stay compatible. Other gated tools use a tool-name scoped key.
    """

    if tool_name == "run_command":
        return (str(args.get("command") or "").strip(), str(args.get("cwd") or ""))
    if tool_name in WRITE_TOOL_NAMES:
        return (tool_name, str(args.get("path") or "").strip())
    if tool_name == "fetch_url":
        parsed = urlparse(str(args.get("url") or ""))
        host = parsed.netloc or str(args.get("url") or "").strip()
        return (tool_name, host.lower())
    if tool_name in DELEGATE_TOOL_NAMES:
        return (tool_name, _short(args.get("task"), 240))
    return (tool_name, _stable_json(args))


def permission_title(tool_name: str, args: dict[str, Any]) -> str:
    """Human-readable ACP permission title."""

    if tool_name == "run_command":
        return f"run_command: {_short(args.get('command')) or '<empty command>'}"
    if tool_name in WRITE_TOOL_NAMES:
        return f"{tool_name}: {_short(args.get('path')) or '<missing path>'}"
    if tool_name == "fetch_url":
        return f"fetch_url: {_short(args.get('url')) or '<missing url>'}"
    if tool_name in DELEGATE_TOOL_NAMES:
        return f"{tool_name} delegate: {_short(args.get('task'), 80) or '<missing task>'}"
    return f"{tool_name}: approve tool call"


def permission_body(tool_name: str, args: dict[str, Any], *, cwd_display: str | None = None) -> str:
    """Human-readable ACP permission body."""

    if tool_name == "run_command":
        command = str(args.get("command") or "").strip() or "<empty command>"
        cwd = str(args.get("cwd") or cwd_display or "")
        return f"Command: {command}\nCWD: {cwd}"
    if tool_name in WRITE_TOOL_NAMES:
        path = str(args.get("path") or "<missing path>")
        expected = str(args.get("expected_sha256") or "")
        lines = [f"Tool: {tool_name}", f"Path: {path}"]
        if expected:
            lines.append(f"Expected SHA-256: {expected}")
        return "\n".join(lines)
    if tool_name == "fetch_url":
        return f"URL: {args.get('url') or '<missing url>'}"
    if tool_name in DELEGATE_TOOL_NAMES:
        task = str(args.get("task") or "<missing task>")
        context = _short(args.get("context"), 300)
        lines = [f"Delegate: {tool_name}", f"Task: {task}"]
        if context:
            lines.append(f"Context: {context}")
        return "\n".join(lines)
    return _stable_json(args)


__all__ = [
    "DELEGATE_TOOL_NAMES",
    "EXECUTE_TOOL_NAMES",
    "NETWORK_TOOL_NAMES",
    "READ_ONLY_TOOL_NAMES",
    "WRITE_TOOL_NAMES",
    "ToolKind",
    "ToolPolicy",
    "ToolRisk",
    "approval_always_option_id",
    "classify_tool",
    "permission_body",
    "permission_cache_key",
    "permission_title",
    "requires_tool_approval",
    "tool_permission_kind",
]
