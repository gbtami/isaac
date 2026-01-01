"""Plan parsing helpers for prompt handling."""

from __future__ import annotations

from typing import Any

from isaac.agent.brain.plan_parser import parse_plan_from_text
from isaac.agent.brain.plan_schema import PlanSteps


def plan_from_planner_result(result_part: Any) -> PlanSteps | None:
    """Parse a planner tool result into PlanSteps."""

    content = getattr(result_part, "content", None)
    plan_obj = content
    if isinstance(plan_obj, dict):
        if plan_obj.get("error"):
            return None
        if "content" in plan_obj:
            plan_obj = plan_obj.get("content")
    if isinstance(plan_obj, PlanSteps) and plan_obj.entries:
        return plan_obj

    if isinstance(plan_obj, str):
        return parse_plan_from_text(plan_obj or "")
    if isinstance(plan_obj, dict):
        return parse_plan_from_text(str(plan_obj))
    return None
