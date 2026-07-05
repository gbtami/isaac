"""Pydantic AI capabilities used by Isaac's ACP runtime.

The goal of this module is to keep Isaac-specific behaviour at Pydantic AI's
extension boundary instead of threading ad-hoc callbacks through every runner.
Capabilities are deliberately small and composable so we can replace more of the
legacy prompt-handler glue incrementally.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
import inspect
import os
from typing import Any, Awaitable, Callable, Iterable

from pydantic_ai import DeferredToolRequests, DeferredToolResults, RunContext, ToolApproved, ToolDenied  # type: ignore
from pydantic_ai.capabilities import AbstractCapability  # type: ignore
from pydantic_ai.tools import ToolDefinition  # type: ignore

ModeGetter = Callable[[], str]
ToolApprovalCallback = Callable[[str, str, dict[str, Any]], Awaitable[bool] | bool]


@dataclass
class ToolModeCapability(AbstractCapability[Any]):
    """Apply Isaac's session mode policy to the visible tool definitions.

    In normal ``ask`` mode, ``run_command`` remains an approval-required tool.
    In ``yolo`` mode, Isaac exposes it as a normal function tool. Pydantic AI v2
    capabilities make this policy part of the assembled agent rather than a
    standalone ``prepare_tools`` constructor callback.
    """

    mode_getter: ModeGetter

    async def prepare_tools(self, ctx: RunContext[Any], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
        _ = ctx
        mode = (self.mode_getter() or "ask").strip().lower()
        if mode != "yolo":
            return tool_defs
        updated: list[ToolDefinition] = []
        for tool_def in tool_defs:
            if tool_def.name == "run_command" and tool_def.kind == "unapproved":
                updated.append(replace(tool_def, kind="function"))
            else:
                updated.append(tool_def)
        return updated


@dataclass
class ACPPermissionCapability(AbstractCapability[Any]):
    """Resolve Pydantic AI deferred approval requests through ACP permissions.

    Pydantic AI v2 can resolve approval-required tool calls inside the same agent
    run via ``handle_deferred_tool_calls``. Isaac still keeps the ACP permission
    UI/policy, but no longer needs to end the model run and manually resume it
    when the provider emits a ``DeferredToolRequests`` output.
    """

    request_tool_approval: ToolApprovalCallback

    async def handle_deferred_tool_calls(
        self,
        ctx: RunContext[Any],
        *,
        requests: DeferredToolRequests,
    ) -> DeferredToolResults | None:
        _ = ctx
        approvals_in = list(getattr(requests, "approvals", []) or [])
        if not approvals_in:
            return None

        approvals: dict[str, ToolApproved | ToolDenied] = {}
        for approval in approvals_in:
            tool_name = str(getattr(approval, "tool_name", "") or "")
            tool_call_id = str(getattr(approval, "tool_call_id", "") or "")
            raw_args = getattr(approval, "args", None)
            args = dict(raw_args) if isinstance(raw_args, dict) else {}
            maybe_allowed = self.request_tool_approval(tool_call_id, tool_name, args)
            allowed = bool(await maybe_allowed) if inspect.isawaitable(maybe_allowed) else bool(maybe_allowed)
            approvals[tool_call_id] = ToolApproved() if allowed else ToolDenied("permission denied")

        return DeferredToolResults(approvals=approvals, metadata=getattr(requests, "metadata", None) or {})


def build_base_capabilities(mode_getter: ModeGetter) -> list[Any]:
    """Capabilities that belong on every Isaac coding agent."""

    capabilities: list[Any] = [ToolModeCapability(mode_getter)]
    capabilities.extend(build_optional_harness_capabilities())
    return capabilities


def build_prompt_capabilities(request_tool_approval: ToolApprovalCallback | None) -> list[Any]:
    """Per-run capabilities assembled from ACP prompt-turn state."""

    if request_tool_approval is None:
        return []
    return [ACPPermissionCapability(request_tool_approval)]


def build_optional_harness_capabilities() -> list[Any]:
    """Load opt-in Harness capabilities without making them mandatory at import time.

    CodeMode is intentionally disabled by default because it changes the tool UX
    substantially by wrapping normal tools behind a ``run_code`` tool. Enable it
    for experiments with ``ISAAC_HARNESS_CODE_MODE=1`` after reviewing approval
    and sandbox behaviour for the target client.
    """

    if os.getenv("ISAAC_HARNESS_CODE_MODE", "").strip().lower() not in {"1", "true", "yes", "on"}:
        return []
    try:
        from pydantic_ai_harness import CodeMode  # type: ignore
    except Exception:
        return []
    return [CodeMode()]


__all__ = [
    "ACPPermissionCapability",
    "ToolModeCapability",
    "build_base_capabilities",
    "build_optional_harness_capabilities",
    "build_prompt_capabilities",
]
