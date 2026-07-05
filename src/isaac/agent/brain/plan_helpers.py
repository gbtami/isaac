"""Plan extraction helpers for prompt handling."""

from __future__ import annotations

from typing import Any

from isaac.agent.brain.plan_schema import PlanSteps


def plan_from_planner_result(result_part: Any) -> PlanSteps | None:
    """Extract a structured planner result from a Pydantic AI tool return part."""

    content = getattr(result_part, "content", None)
    if isinstance(content, dict):
        if content.get("error"):
            return None
        content = content.get("content")
    if isinstance(content, PlanSteps) and content.entries:
        return content
    return None
