"""Tool policy helpers for pydantic-ai agent runners."""

from __future__ import annotations

from dataclasses import replace
from typing import Awaitable, Callable

from pydantic_ai import RunContext  # type: ignore
from pydantic_ai.tools import ToolDefinition  # type: ignore

_ModeGetter = Callable[[], str]


def build_prepare_tools_for_mode(
    mode_getter: _ModeGetter,
) -> Callable[[RunContext[object], list[ToolDefinition]], Awaitable[list[ToolDefinition]]]:
    """Build a tool-prepare hook that toggles run_command approvals by mode.

    Ask mode keeps `run_command` as an approval-required tool; yolo mode promotes
    it to a normal function tool.
    """

    async def _prepare(_ctx: RunContext[object], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
        mode = (mode_getter() or "ask").strip().lower()
        if mode != "yolo":
            return tool_defs
        updated: list[ToolDefinition] = []
        for tool_def in tool_defs:
            if tool_def.name == "run_command" and tool_def.kind == "unapproved":
                updated.append(replace(tool_def, kind="function"))
            else:
                updated.append(tool_def)
        return updated

    return _prepare
