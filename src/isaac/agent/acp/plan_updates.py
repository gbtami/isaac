"""ACP plan update helpers for PlanSteps inputs."""

from __future__ import annotations

from typing import Any, Sequence

from acp.helpers import update_plan
from acp.schema import AgentPlanContentUpdate, AgentPlanRemovedUpdate, PlanEntry, PlanUpdateItems

from isaac.agent.brain.plan_schema import PlanSteps, PlanStep

DEFAULT_PLAN_ID = "isaac-plan"


def _plan_entries(
    plan_steps: PlanSteps,
    *,
    active_index: int | None,
    status_all: str | None,
    statuses: Sequence[str] | None,
) -> list[PlanEntry]:
    entries: list[PlanEntry] = []
    for idx, step in enumerate(plan_steps.entries):
        content = step.content.strip()
        if not content:
            continue
        priority = step.priority if step.priority in {"high", "medium", "low"} else "medium"
        if statuses is not None and idx < len(statuses):
            status = statuses[idx]
        elif status_all is not None:
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
        step_id = step.id or f"step_{idx + 1}"
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
    use_incremental: bool = False,
    plan_id: str = DEFAULT_PLAN_ID,
    statuses: Sequence[str] | None = None,
) -> Any | None:
    """Convert PlanSteps into an ACP plan update with status metadata.

    ACP 0.11 adds granular ``plan_update`` session updates. Isaac emits those
    when the connected client advertises plan-update support, and falls back to
    the legacy full-plan ``plan`` update otherwise.
    """

    if not isinstance(plan_steps, PlanSteps):
        return None
    entries = _plan_entries(plan_steps, active_index=active_index, status_all=status_all, statuses=statuses)
    if not entries:
        return None
    try:
        if use_incremental:
            return AgentPlanContentUpdate(
                session_update="plan_update",
                plan=PlanUpdateItems(type="items", id=plan_id, entries=entries),
            )
        return update_plan(entries)
    except Exception:
        return None


def build_plan_removed(*, plan_id: str = DEFAULT_PLAN_ID) -> AgentPlanRemovedUpdate:
    """Build an ACP 0.11 granular plan-removal update."""

    return AgentPlanRemovedUpdate(session_update="plan_removed", id=plan_id)


def plan_steps_from_entries(steps: list[PlanStep]) -> PlanSteps:
    """Create PlanSteps from existing entries."""

    return PlanSteps(entries=steps)


__all__ = ["DEFAULT_PLAN_ID", "build_plan_removed", "build_plan_update", "plan_steps_from_entries"]
