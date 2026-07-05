"""Structured plan schemas used by prompt handling."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class PlanStep(BaseModel):
    content: str
    priority: Literal["high", "medium", "low"] = "medium"
    id: str | None = None


class PlanSteps(BaseModel):
    entries: list[PlanStep]


__all__ = ["PlanStep", "PlanSteps"]
