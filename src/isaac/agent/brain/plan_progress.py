"""Run-scoped plan progress state for ACP plan updates.

Planner delegates produce the ordered steps, but ordinary file/search/command
calls should not implicitly complete those steps.  This module keeps plan status
changes explicit and stable for the duration of a prompt turn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, cast

from isaac.agent.brain.plan_schema import PlanStep, PlanSteps

PlanStatus = Literal["pending", "in_progress", "completed"]
_VALID_STATUSES: set[str] = {"pending", "in_progress", "completed"}


def normalize_plan_steps(plan: PlanSteps) -> PlanSteps:
    """Return a copy of ``plan`` with stable deterministic step ids."""

    normalized: list[PlanStep] = []
    used: set[str] = set()
    for index, step in enumerate(plan.entries):
        step_id = (step.id or "").strip() or f"step_{index + 1}"
        if step_id in used:
            step_id = f"step_{index + 1}"
        used.add(step_id)
        normalized.append(step.model_copy(update={"id": step_id}))
    return PlanSteps(entries=normalized)


@dataclass
class PlanProgress:
    """Mutable status tracker for one model run's plan."""

    plan: PlanSteps | None = None
    statuses: list[PlanStatus] = field(default_factory=list)

    def set_plan(self, plan: PlanSteps) -> PlanSteps:
        """Install a freshly emitted plan and mark only the first step active."""

        self.plan = normalize_plan_steps(plan)
        self.statuses = ["pending" for _ in self.plan.entries]
        if self.statuses:
            self.statuses[0] = "in_progress"
        return self.plan

    @property
    def has_plan(self) -> bool:
        return bool(self.plan and self.plan.entries)

    @property
    def active_index(self) -> int | None:
        for index, status in enumerate(self.statuses):
            if status == "in_progress":
                return index
        return None

    @property
    def is_completed(self) -> bool:
        return bool(self.statuses) and all(status == "completed" for status in self.statuses)

    def mark(self, step: int | str, status: str, note: str | None = None) -> dict[str, object]:
        """Explicitly update one plan step.

        ``step`` may be a 1-based numeric index or a stable step id.  Completing
        a step activates the next pending step, but only because the model made
        an explicit completion call first.
        """

        if not self.plan:
            return {"content": "No active plan to update.", "error": "No active plan"}
        normalized_status = status.strip().lower()
        if normalized_status not in _VALID_STATUSES:
            return {
                "content": "Invalid plan step status.",
                "error": f"Invalid status: {status}",
                "valid_statuses": sorted(_VALID_STATUSES),
            }
        index = self._resolve_step_index(step)
        if index is None:
            return {"content": "Unknown plan step.", "error": f"Unknown plan step: {step}"}

        typed_status = cast(PlanStatus, normalized_status)
        self.statuses[index] = typed_status
        if typed_status == "in_progress":
            self._single_active_step(index)
        elif typed_status == "completed":
            self._activate_next_pending(after=index)

        step_obj = self.plan.entries[index]
        message = f"Marked plan step {index + 1} ({step_obj.id}) as {typed_status}."
        if note:
            message = f"{message} Note: {note.strip()}"
        return {
            "content": message,
            "step": index + 1,
            "step_id": step_obj.id,
            "status": typed_status,
            "note": note,
            "plan_completed": self.is_completed,
            "statuses": list(self.statuses),
        }

    def _resolve_step_index(self, step: int | str) -> int | None:
        if isinstance(step, int):
            index = step - 1
            return index if 0 <= index < len(self.statuses) else None
        step_text = str(step).strip()
        if step_text.isdigit():
            return self._resolve_step_index(int(step_text))
        if self.plan is None:
            return None
        for index, plan_step in enumerate(self.plan.entries):
            if plan_step.id == step_text:
                return index
        return None

    def _single_active_step(self, active_index: int) -> None:
        for index, status in enumerate(self.statuses):
            if index != active_index and status == "in_progress":
                self.statuses[index] = "pending"

    def _activate_next_pending(self, *, after: int) -> None:
        if any(status == "in_progress" for status in self.statuses):
            return
        for index in range(after + 1, len(self.statuses)):
            if self.statuses[index] == "pending":
                self.statuses[index] = "in_progress"
                return


__all__ = ["PlanProgress", "PlanStatus", "normalize_plan_steps"]
