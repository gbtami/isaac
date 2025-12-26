"""Planner delegate tool implementation."""

from __future__ import annotations

from pydantic_ai import RunContext

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
- You may consult other delegate tools to clarify context, but do not ask them to implement changes.
- If context is missing, make reasonable assumptions and proceed.
Return only the JSON plan.
"""

PLANNER_TOOL_INSTRUCTIONS = """
Return a short plan in the required JSON shape.
- Keep steps specific and ordered.
- Use priority="medium" when unsure.
"""

PLANNER_TOOL_SPEC = DelegateToolSpec(
    name="planner",
    description="Delegate planning to a specialized planner agent.",
    instructions=PLANNER_TOOL_INSTRUCTIONS,
    system_prompt=PLANNER_SYSTEM_PROMPT,
    tool_names=_READ_ONLY_TOOL_NAMES,
    output_type=PlanSteps,
    log_context="delegate_planner",
)


async def planner(ctx: RunContext, task: str, context: str | None = None) -> dict[str, object]:
    """Delegate planning to a specialized planner agent."""
    _ = ctx
    result = await run_delegate_tool(
        PLANNER_TOOL_SPEC,
        task=task,
        context=context,
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
