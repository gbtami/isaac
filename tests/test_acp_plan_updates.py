from __future__ import annotations

from acp.schema import AgentPlanContentUpdate, AgentPlanRemovedUpdate, AgentPlanUpdate

from isaac.agent.acp.plan_updates import DEFAULT_PLAN_ID, build_plan_removed, build_plan_update
from isaac.agent.brain.plan_schema import PlanStep, PlanSteps


def test_build_plan_update_sets_statuses_and_ids() -> None:
    steps = PlanSteps(
        entries=[
            PlanStep(content="one"),
            PlanStep(content="two", id="step-two"),
            PlanStep(content="three"),
        ]
    )

    update = build_plan_update(steps, active_index=1, status_all=None)

    assert isinstance(update, AgentPlanUpdate)
    assert [entry.status for entry in update.entries] == ["completed", "in_progress", "pending"]
    assert update.entries[0].field_meta.get("id")
    assert update.entries[1].field_meta.get("id") == "step-two"


def test_build_plan_update_status_all_completed() -> None:
    steps = PlanSteps(entries=[PlanStep(content="alpha"), PlanStep(content="beta")])

    update = build_plan_update(steps, active_index=None, status_all="completed")

    assert update is not None
    assert all(entry.status == "completed" for entry in update.entries)


def test_build_plan_update_uses_granular_update_when_requested() -> None:
    steps = PlanSteps(entries=[PlanStep(content="alpha"), PlanStep(content="beta")])

    update = build_plan_update(steps, active_index=0, status_all=None, use_incremental=True)

    assert isinstance(update, AgentPlanContentUpdate)
    assert update.plan.id == DEFAULT_PLAN_ID
    assert update.plan.type == "items"
    assert [entry.status for entry in update.plan.entries] == ["in_progress", "pending"]


def test_build_plan_removed_targets_default_plan() -> None:
    update = build_plan_removed()

    assert isinstance(update, AgentPlanRemovedUpdate)
    assert update.id == DEFAULT_PLAN_ID
