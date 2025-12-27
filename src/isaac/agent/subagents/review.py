"""Review delegate tool implementation."""

from __future__ import annotations

from pydantic_ai import RunContext

from isaac.agent.subagents.args import ReviewArgs
from isaac.agent.subagents.delegate_tools import DelegateToolSpec, register_delegate_tool, run_delegate_tool
from isaac.agent.subagents.outputs import ReviewDelegateOutput

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
- You may consult the planner delegate for structure, but still return a review summary yourself.
- If referencing code, include file paths and line numbers when available.
- When reporting issues, prefix bullets with severity when possible (high/medium/low).
- If nothing stands out, respond with "No issues found."

Output:
- Return ONLY JSON with keys: summary, findings, tests.
- findings is a list of objects with: severity, description, file, line, suggestion.
- If there are no issues, set summary="No issues found." and findings=[].
"""

REVIEW_TOOL_INSTRUCTIONS = """
Return a short review in JSON.
- summary: one sentence.
- findings: list of issues with severity, file, line, suggestion when possible.
- tests: list of missing or suggested tests.
"""


def _review_summary(output: object) -> str | None:
    """Extract the review summary for carryover context."""

    if isinstance(output, ReviewDelegateOutput):
        return output.summary
    return None


REVIEW_TOOL_SPEC = DelegateToolSpec(
    name="review",
    description="Delegate a focused review of code or output.",
    instructions=REVIEW_TOOL_INSTRUCTIONS,
    system_prompt=REVIEW_SYSTEM_PROMPT,
    tool_names=_READ_ONLY_TOOL_NAMES + ("planner",),
    log_context="delegate_review",
    output_type=ReviewDelegateOutput,
    summary_extractor=_review_summary,
    min_summary_chars=60,
    continuation_prompt=(
        "Your previous response was too brief. Expand the summary and findings while keeping the JSON shape."
    ),
)


async def review(
    ctx: RunContext,
    task: str,
    context: str | None = None,
    session_id: str | None = None,
    carryover: bool = False,
) -> dict[str, object]:
    """Delegate a focused review to a specialized agent."""
    _ = ctx
    return await run_delegate_tool(
        REVIEW_TOOL_SPEC,
        task=task,
        context=context,
        session_id=session_id,
        carryover=carryover,
    )


register_delegate_tool(REVIEW_TOOL_SPEC, handler=review, arg_model=ReviewArgs)
