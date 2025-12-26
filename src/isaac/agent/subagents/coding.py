"""Coding delegate tool implementation."""

from __future__ import annotations

from pydantic_ai import RunContext

from isaac.agent.subagents.args import CodingArgs
from isaac.agent.subagents.delegate_tools import DelegateToolSpec, register_delegate_tool, run_delegate_tool

CODING_SYSTEM_PROMPT = """
You are Isaac's coding delegate.
You receive only the task and optional context provided by the caller.
Your job is to implement the requested change with minimal, precise edits.

Guidelines:
- Inspect files before editing; preserve existing style and patterns.
- Use edit_file or apply_patch for changes; avoid unnecessary rewrites.
- You may use read-only tools to understand the codebase.
- Avoid run_command unless explicitly asked by the task.
 - You may delegate to other tools (planner/review) for planning or validation, but keep depth low.

After acting, return a concise summary and list files changed.
If you did not change files, explain why.
"""

CODING_TOOL_INSTRUCTIONS = """
Implement the requested change.
- Be specific and keep output short.
- Mention any follow-up work or risks.
"""

_CODING_TOOL_NAMES: tuple[str, ...] = (
    "list_files",
    "read_file",
    "file_summary",
    "code_search",
    "fetch_url",
    "edit_file",
    "apply_patch",
)

CODING_TOOL_SPEC = DelegateToolSpec(
    name="coding",
    description="Delegate implementation work to a coding-focused agent.",
    instructions=CODING_TOOL_INSTRUCTIONS,
    system_prompt=CODING_SYSTEM_PROMPT,
    tool_names=_CODING_TOOL_NAMES,
    log_context="delegate_coding",
)


async def coding(ctx: RunContext, task: str, context: str | None = None) -> dict[str, object]:
    """Delegate implementation work to a coding-focused agent."""
    _ = ctx
    return await run_delegate_tool(
        CODING_TOOL_SPEC,
        task=task,
        context=context,
    )


register_delegate_tool(CODING_TOOL_SPEC, handler=coding, arg_model=CodingArgs)
