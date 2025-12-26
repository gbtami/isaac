"""Helpers for building and updating ACP plan updates."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from acp.helpers import update_plan
from acp.schema import PlanEntry


def plan_with_status(plan_update: Any, *, active_index: int | None = None, status_all: str | None = None) -> Any:
    try:
        entries = list(getattr(plan_update, "entries", []) or [])
    except Exception:
        return plan_update
    if not entries:
        return plan_update
    updated: list[PlanEntry] = []
    for idx, entry in enumerate(entries):
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
            status = getattr(entry, "status", "pending")
        try:
            updated.append(entry.model_copy(update={"status": status}))
        except Exception:
            updated.append(entry)
    try:
        return update_plan(updated)
    except Exception:
        return plan_update


def plan_update_from_steps(steps: list[Any]) -> Any | None:
    """Build a plan update with stable IDs from structured PlanStep objects."""

    entries: list[PlanEntry] = []
    for idx, item in enumerate(steps):
        content = getattr(item, "content", "")
        if not isinstance(content, str) or not content.strip():
            continue
        priority = getattr(item, "priority", "medium")
        if priority not in {"high", "medium", "low"}:
            priority = "medium"
        step_id = getattr(item, "id", None) or f"step_{idx + 1}_{uuid4().hex[:6]}"
        try:
            entry = PlanEntry(content=content.strip(), priority=priority, status="pending")
            entry = entry.model_copy(update={"field_meta": {"id": step_id}})
            entries.append(entry)
        except Exception:
            continue
    if not entries:
        return None
    try:
        return update_plan(entries)
    except Exception:
        return None
