"""Coding delegate tool implementation."""

from __future__ import annotations

from pydantic_ai import RunContext

from isaac.agent.subagents.args import CodingArgs
from isaac.agent.subagents.delegate_tools import DelegateToolSpec, register_delegate_tool, run_delegate_tool
from isaac.agent.subagents.outputs import CodingDelegateOutput

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
- When reporting results, include a short "Files:" list with each file and intent.

After acting, return a concise summary and list files changed.
If you did not change files, explain why.

Output:
- Return ONLY JSON with keys: summary, files, tests, risks, followups.
- files is a list of objects with: path, summary, intent.
- If no files changed, use files=[] and explain why in summary.
"""

CODING_TOOL_INSTRUCTIONS = """
Implement the requested change.
- Be specific and keep output short.
- Mention any follow-up work or risks.
- Output JSON only, matching the required keys.
"""

_CODING_TOOL_NAMES: tuple[str, ...] = (
    "list_files",
    "read_file",
    "file_summary",
    "code_search",
    "fetch_url",
    "edit_file",
    "apply_patch",
    "planner",
    "review",
)


def _coding_summary(output: object) -> str | None:
    """Extract the coding summary for carryover context."""

    if isinstance(output, CodingDelegateOutput):
        return output.summary
    return None


CODING_TOOL_SPEC = DelegateToolSpec(
    name="coding",
    description="Delegate implementation work to a coding-focused agent.",
    instructions=CODING_TOOL_INSTRUCTIONS,
    system_prompt=CODING_SYSTEM_PROMPT,
    tool_names=_CODING_TOOL_NAMES,
    log_context="delegate_coding",
    output_type=CodingDelegateOutput,
    summary_extractor=_coding_summary,
    min_summary_chars=80,
    continuation_prompt=(
        "Your previous response was too brief. Expand the summary and list "
        "files/tests/risks/followups while keeping the JSON shape."
    ),
)


async def coding(
    ctx: RunContext,
    task: str,
    context: str | None = None,
    session_id: str | None = None,
    carryover: bool = False,
) -> dict[str, object]:
    """Delegate implementation work to a coding-focused agent."""
    return await run_delegate_tool(
        CODING_TOOL_SPEC,
        task=task,
        context=context,
        session_id=session_id,
        carryover=carryover,
        tool_call_id=getattr(ctx, "tool_call_id", None),
    )


register_delegate_tool(CODING_TOOL_SPEC, handler=coding, arg_model=CodingArgs)
