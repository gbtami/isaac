from __future__ import annotations

from acp.schema import AgentPlanUpdate

from isaac.agent.acp.plan_updates import build_plan_update
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
