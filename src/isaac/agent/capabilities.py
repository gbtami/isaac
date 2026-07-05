"""Pydantic AI capabilities used by Isaac's ACP runtime.

The goal of this module is to keep Isaac-specific behaviour at Pydantic AI's
extension boundary instead of threading ad-hoc callbacks through every runner.
Capabilities are deliberately assembled as small, composable built-ins so ACP
runtime policy stays at Pydantic AI's extension boundary.
"""

from __future__ import annotations

import copy
from dataclasses import is_dataclass, replace
import inspect
import os
from typing import Any, Awaitable, Callable

from pydantic_ai import DeferredToolRequests, DeferredToolResults, RunContext, ToolDenied  # type: ignore
from pydantic_ai.capabilities import (  # type: ignore
    Capability,
    HandleDeferredToolCalls,
    PrefixTools,
    PrepareTools,
    ProcessHistory,
    ReinjectSystemPrompt,
    Toolset as ToolsetCapability,
)
from pydantic_ai.tools import ToolDefinition  # type: ignore

from isaac.agent.brain.history_processors import sanitize_message_history
from isaac.agent.brain.recent_files import recent_files_context_text

ModeGetter = Callable[[], str]
ToolApprovalCallback = Callable[[str, str, dict[str, Any]], Awaitable[bool] | bool]


def _copy_tool_definition(tool_def: ToolDefinition, **updates: Any) -> ToolDefinition:
    """Return a ToolDefinition copy with ``updates`` applied.

    Pydantic AI's public docs treat ``ToolDefinition`` as the object passed to
    tool-preparation hooks, but implementation details have changed across the
    fast-moving v2 releases. Keep this helper defensive so Isaac's mode policy
    does not depend on whether the object is currently a dataclass, pydantic
    model, attrs object, or plain class.
    """

    model_copy = getattr(tool_def, "model_copy", None)
    if callable(model_copy):
        return model_copy(update=updates)
    if is_dataclass(tool_def):
        return replace(tool_def, **updates)
    copied = copy.copy(tool_def)
    for key, value in updates.items():
        setattr(copied, key, value)
    return copied


async def _prepare_tools_for_mode(
    mode_getter: ModeGetter,
    ctx: RunContext[Any],
    tool_defs: list[ToolDefinition],
) -> list[ToolDefinition]:
    """Apply Isaac's session mode policy to visible tool definitions.

    In normal ``ask`` mode, ``run_command`` remains an approval-required tool.
    In ``yolo`` mode, Isaac exposes it as a normal function tool. This is now
    plugged into Pydantic AI through the built-in ``PrepareTools`` capability
    instead of the older Agent constructor hook.
    """

    _ = ctx
    mode = (mode_getter() or "ask").strip().lower()
    if mode != "yolo":
        return tool_defs

    updated: list[ToolDefinition] = []
    for tool_def in tool_defs:
        if getattr(tool_def, "name", None) == "run_command" and getattr(tool_def, "kind", None) == "unapproved":
            updated.append(_copy_tool_definition(tool_def, kind="function"))
        else:
            updated.append(tool_def)
    return updated


async def build_acp_deferred_tool_results(
    requests: DeferredToolRequests,
    request_tool_approval: ToolApprovalCallback,
) -> DeferredToolResults | None:
    """Resolve Pydantic AI deferred approval requests through Isaac's ACP policy.

    This helper is shared by the inline capability path and the direct
    deferred-output safety path in ``stream_with_runner`` so approval semantics
    stay identical whichever event shape a runner emits.
    """

    approvals_in = list(getattr(requests, "approvals", []) or [])
    if not approvals_in:
        return None

    approvals: dict[str, bool | ToolDenied] = {}
    for approval in approvals_in:
        tool_name = str(getattr(approval, "tool_name", "") or "")
        tool_call_id = str(getattr(approval, "tool_call_id", "") or "")
        raw_args = getattr(approval, "args", None)
        args = dict(raw_args) if isinstance(raw_args, dict) else {}
        maybe_allowed = request_tool_approval(tool_call_id, tool_name, args)
        allowed = bool(await maybe_allowed) if inspect.isawaitable(maybe_allowed) else bool(maybe_allowed)
        approvals[tool_call_id] = True if allowed else ToolDenied("permission denied")

    return requests.build_results(approvals=approvals, metadata=getattr(requests, "metadata", None) or {})


def build_mode_capability(mode_getter: ModeGetter) -> Any:
    """Build the capability that maps Isaac modes to Pydantic AI tools."""

    async def prepare_tools(ctx: RunContext[Any], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
        return await _prepare_tools_for_mode(mode_getter, ctx, tool_defs)

    return PrepareTools(prepare_tools)


def build_acp_permission_capability(request_tool_approval: ToolApprovalCallback) -> Any:
    """Build the capability that resolves deferred approvals via ACP."""

    async def handle_deferred(ctx: RunContext[Any], requests: DeferredToolRequests) -> DeferredToolResults | None:
        _ = ctx
        return await build_acp_deferred_tool_results(requests, request_tool_approval)

    return HandleDeferredToolCalls(handle_deferred)


def build_system_prompt_capability() -> Any:
    """Build the capability that keeps Isaac's server-side system prompt authoritative.

    ACP clients and persisted UI history may omit system prompts entirely, or may
    round-trip an older system prompt from before a session cwd/model switch.
    Pydantic AI v2 ships this as a first-class capability, so Isaac no longer
    needs to rely on the converted chat history carrying the correct prompt.
    """

    return ReinjectSystemPrompt(replace_existing=True)


def build_history_sanitizer_capability() -> Any:
    """Build the capability that cleans provider-bound message history.

    This keeps Isaac's provider-safety history processor on Pydantic AI's
    capability hook instead of relying on removed/constructor-level history
    processor plumbing. The processor itself stays Isaac-owned because it
    preserves pydantic-ai message metadata while dropping empty text parts that
    several providers reject.
    """

    return ProcessHistory(sanitize_message_history)


def build_recent_files_capability(recent_files: list[str], context_count: int) -> Any | None:
    """Build a per-run instruction capability for ambiguous file follow-ups.

    Isaac tracks files touched by mutating tools so short follow-up prompts like
    "open it again" or "fix that file" can be grounded. Pydantic AI v2 lets us
    express this as transient run instructions, so prompt handling no longer has
    to mutate the persisted chat history just to provide runtime context.
    """

    message = recent_files_context_text(recent_files, context_count)
    if message is None:
        return None
    return Capability(
        instructions=message,
        id="isaac-recent-files",
        description="Transient context about files touched earlier in this session.",
    )


def build_base_capabilities(mode_getter: ModeGetter) -> list[Any]:
    """Capabilities that belong on every Isaac coding agent."""

    capabilities: list[Any] = [
        build_system_prompt_capability(),
        build_history_sanitizer_capability(),
        build_mode_capability(mode_getter),
    ]
    capabilities.extend(build_optional_harness_capabilities())
    return capabilities


def build_prompt_capabilities(request_tool_approval: ToolApprovalCallback | None) -> list[Any]:
    """Per-run capabilities assembled from ACP prompt-turn state."""

    if request_tool_approval is None:
        return []
    return [build_acp_permission_capability(request_tool_approval)]


_ENV_TRUE = {"1", "true", "yes", "on"}


def _env_enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in _ENV_TRUE


def build_optional_harness_capabilities() -> list[Any]:
    """Load opt-in Harness capabilities without making them a base dependency.

    Harness integration is intentionally experimental and opt-in. FileSystem and
    Shell are prefixed as ``harness_*`` tools so they cannot change Isaac's
    existing ACP-visible tool contract or collide with Isaac's own ``read_file``,
    ``edit_file``, and ``run_command`` tools. CodeMode remains disabled by
    default because it changes the tool UX substantially by wrapping normal tools
    behind a ``run_code`` tool.
    """

    capabilities: list[Any] = []
    if _env_enabled("ISAAC_HARNESS_FILESYSTEM"):
        try:
            from pydantic_ai_harness import FileSystem  # type: ignore
        except Exception:
            pass
        else:
            capabilities.append(PrefixTools(ToolsetCapability(FileSystem()), prefix="harness"))

    if _env_enabled("ISAAC_HARNESS_SHELL"):
        try:
            from pydantic_ai_harness import Shell  # type: ignore
        except Exception:
            pass
        else:
            capabilities.append(PrefixTools(ToolsetCapability(Shell()), prefix="harness"))

    if _env_enabled("ISAAC_HARNESS_CODE_MODE"):
        try:
            from pydantic_ai_harness import CodeMode  # type: ignore
        except Exception:
            pass
        else:
            capabilities.append(CodeMode())

    return capabilities


__all__ = [
    "build_acp_deferred_tool_results",
    "build_acp_permission_capability",
    "build_base_capabilities",
    "build_history_sanitizer_capability",
    "build_mode_capability",
    "build_optional_harness_capabilities",
    "build_recent_files_capability",
    "build_prompt_capabilities",
    "build_system_prompt_capability",
]
