"""Argument models for delegate sub-agent tools."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class DelegateAgentArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")

    task: str = Field(
        ...,
        min_length=1,
        description="Task or prompt for the delegate agent.",
    )
    context: str | None = Field(
        None,
        description="Optional extra context to include with the task.",
    )


class PlannerArgs(DelegateAgentArgs):
    pass


class ReviewArgs(DelegateAgentArgs):
    pass


class CodingArgs(DelegateAgentArgs):
    pass
