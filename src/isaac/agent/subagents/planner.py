"""Planner delegate tool implementation."""

from __future__ import annotations

from typing import Any

from isaac.agent.ai_types import ToolContext

from isaac.agent.brain.plan_schema import PlanSteps
from isaac.agent.brain.plan_parser import parse_plan_from_text
from isaac.agent.subagents.args import PlannerArgs
from isaac.agent.subagents.delegate_tools import DelegateToolSpec, register_delegate_tool, run_delegate_tool

_READ_ONLY_TOOL_NAMES: tuple[str, ...] = (
    "list_files",
    "read_file",
    "file_summary",
    "code_search",
    "fetch_url",
)

PLANNER_SYSTEM_PROMPT = """
You are Isaac's planning delegate.
You receive only the task and optional context provided by the caller.
Your job is to produce a concise, actionable plan for the main agent to execute.

Rules:
- Output only JSON matching {"entries":[{"content":"...", "priority":"high|medium|low"}]}.
- Use 3-6 steps unless the task is trivial (1-2 steps).
- Each step is an outcome, not a command list.
- No code, no execution, no extra commentary.
- Do not call other delegate tools; rely on read-only tools only.
- If you consult tools, still return only the JSON plan with no extra fields or prose.
- If context is missing, make reasonable assumptions and proceed.
Return only the JSON plan.
"""

PLANNER_TOOL_INSTRUCTIONS = """
Return a short plan in the required JSON shape.
- Keep steps specific and ordered.
- Use priority="medium" when unsure.
"""


def _planner_summary(output: object) -> str | None:
    """Summarize planner output for optional carryover context."""

    if isinstance(output, PlanSteps):
        return "; ".join(step.content for step in output.entries)
    return None


PLANNER_TOOL_SPEC = DelegateToolSpec(
    name="planner",
    description="Delegate planning to a specialized planner agent.",
    instructions=PLANNER_TOOL_INSTRUCTIONS,
    system_prompt=PLANNER_SYSTEM_PROMPT,
    tool_names=_READ_ONLY_TOOL_NAMES,
    output_type=PlanSteps,
    log_context="delegate_planner",
    summary_extractor=_planner_summary,
)


async def planner(
    ctx: ToolContext,
    task: str,
    context: str | None = None,
    session_id: str | None = None,
    carryover: bool = False,
) -> dict[str, Any]:
    """Delegate planning to a specialized planner agent."""
    result = await run_delegate_tool(
        PLANNER_TOOL_SPEC,
        task=task,
        context=context,
        session_id=session_id,
        carryover=carryover,
        tool_call_id=getattr(ctx, "tool_call_id", None),
    )
    if result.get("error"):
        return result
    content = result.get("content")
    if isinstance(content, PlanSteps):
        return result
    if isinstance(content, str):
        parsed = parse_plan_from_text(content)
        if parsed is not None:
            return {"content": parsed, "error": None}
    return result


register_delegate_tool(PLANNER_TOOL_SPEC, handler=planner, arg_model=PlannerArgs)
