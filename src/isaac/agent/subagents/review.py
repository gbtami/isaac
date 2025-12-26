"""Review delegate tool implementation."""

from __future__ import annotations

from pydantic_ai import RunContext

from isaac.agent.subagents.args import ReviewArgs
from isaac.agent.subagents.delegate_tools import DelegateToolSpec, register_delegate_tool, run_delegate_tool

_READ_ONLY_TOOL_NAMES: tuple[str, ...] = (
    "list_files",
    "read_file",
    "file_summary",
    "code_search",
    "fetch_url",
)

REVIEW_SYSTEM_PROMPT = """
You are Isaac's review delegate.
You receive only the task and optional context provided by the caller.
Your job is to review code or output for correctness, edge cases, and missing tests.

Guidelines:
- Focus on high-signal issues; skip style nits.
- Be concise and specific.
- Use tools to inspect files if needed; do not edit files directly or run commands.
- You may ask other delegate tools for help (for example, to apply fixes), but still return a review summary yourself.
- If referencing code, include file paths and line numbers when available.
- If nothing stands out, respond with "No issues found."
"""

REVIEW_TOOL_INSTRUCTIONS = """
Return a short review with clear, actionable points.
- Use brief bullets.
- Note severity when relevant (high/medium/low).
"""

REVIEW_TOOL_SPEC = DelegateToolSpec(
    name="review",
    description="Delegate a focused review of code or output.",
    instructions=REVIEW_TOOL_INSTRUCTIONS,
    system_prompt=REVIEW_SYSTEM_PROMPT,
    tool_names=_READ_ONLY_TOOL_NAMES + ("planner",),
    log_context="delegate_review",
)


async def review(ctx: RunContext, task: str, context: str | None = None) -> dict[str, object]:
    """Delegate a focused review to a specialized agent."""
    _ = ctx
    return await run_delegate_tool(
        REVIEW_TOOL_SPEC,
        task=task,
        context=context,
    )


register_delegate_tool(REVIEW_TOOL_SPEC, handler=review, arg_model=ReviewArgs)
