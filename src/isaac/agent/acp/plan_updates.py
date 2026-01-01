"""ACP plan update helpers for PlanSteps inputs."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from acp.helpers import update_plan
from acp.schema import PlanEntry

from isaac.agent.brain.plan_schema import PlanSteps, PlanStep


def _plan_entries(plan_steps: PlanSteps, *, active_index: int | None, status_all: str | None) -> list[PlanEntry]:
    entries: list[PlanEntry] = []
    for idx, step in enumerate(plan_steps.entries):
        content = step.content.strip()
        if not content:
            continue
        priority = step.priority if step.priority in {"high", "medium", "low"} else "medium"
        if status_all is not None:
            status = status_all
        elif active_index is not None:
            if idx < active_index:
                status = "completed"
            elif idx == active_index:
                status = "in_progress"
            else:
                status = "pending"
        else:
            status = "pending"
        entry = PlanEntry(content=content, priority=priority, status=status)
        step_id = step.id or f"step_{idx + 1}_{uuid4().hex[:6]}"
        try:
            entry = entry.model_copy(update={"field_meta": {"id": step_id}})
        except Exception:
            pass
        entries.append(entry)
    return entries


def build_plan_update(
    plan_steps: Any,
    *,
    active_index: int | None = None,
    status_all: str | None = None,
) -> Any | None:
    """Convert PlanSteps into an ACP update with status metadata."""

    if not isinstance(plan_steps, PlanSteps):
        return None
    entries = _plan_entries(plan_steps, active_index=active_index, status_all=status_all)
    if not entries:
        return None
    try:
        return update_plan(entries)
    except Exception:
        return None


def plan_steps_from_entries(steps: list[PlanStep]) -> PlanSteps:
    """Create PlanSteps from existing entries."""

    return PlanSteps(entries=steps)


__all__ = ["build_plan_update", "plan_steps_from_entries"]
