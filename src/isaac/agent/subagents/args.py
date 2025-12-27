"""Argument models for delegate sub-agent tools."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class DelegateAgentArgs(BaseModel):
    """Base arguments shared by delegate sub-agent tools."""

    model_config = ConfigDict(extra="ignore")

    task: str = Field(
        ...,
        min_length=1,
        description="Task or prompt for the delegate agent.",
    )
    context: str | None = Field(
        None,
        description="Optional extra context to include with the task. This is the only context shared.",
    )
    session_id: str | None = Field(
        None,
        description=(
            "Optional delegate session id for multi-turn follow-ups. "
            "Reusing the same id lets the delegate carry over its own summary only."
        ),
    )
    carryover: bool = Field(
        False,
        description=(
            "Whether to include the previous delegate summary for this session id. "
            "Defaults to false to keep delegate runs isolated."
        ),
    )


class PlannerArgs(DelegateAgentArgs):
    pass


class ReviewArgs(DelegateAgentArgs):
    pass


class CodingArgs(DelegateAgentArgs):
    pass
