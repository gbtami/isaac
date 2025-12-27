"""Structured output schemas for delegate sub-agent tools.

These models intentionally keep required fields minimal while still providing a
consistent shape that downstream consumers (the main agent) can depend on.
Extra fields are ignored so delegate agents can add detail without breaking
validation.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DelegateFileChange(BaseModel):
    """Describe a single file change or inspection made by a delegate."""

    model_config = ConfigDict(extra="ignore")

    path: str = Field(..., description="Path to the file that was changed or inspected.")
    summary: str = Field(..., description="Short description of what changed or was verified.")
    intent: str | None = Field(
        None,
        description="Optional rationale or goal behind the change.",
    )


class CodingDelegateOutput(BaseModel):
    """Structured response for coding delegate runs."""

    model_config = ConfigDict(extra="ignore")

    summary: str = Field(..., description="Concise summary of work completed or skipped.")
    files: list[DelegateFileChange] = Field(
        default_factory=list,
        description="Files touched or reviewed during the delegate run.",
    )
    tests: list[str] = Field(
        default_factory=list,
        description="Tests run or suggested for follow-up.",
    )
    risks: list[str] = Field(
        default_factory=list,
        description="Potential risks, regressions, or open concerns.",
    )
    followups: list[str] = Field(
        default_factory=list,
        description="Recommended next steps for the main agent.",
    )


class ReviewFinding(BaseModel):
    """A single review finding with optional location metadata."""

    model_config = ConfigDict(extra="ignore")

    severity: Literal["high", "medium", "low"] | None = Field(
        None,
        description="Severity of the finding when known.",
    )
    description: str = Field(..., description="Issue description and why it matters.")
    file: str | None = Field(
        None,
        description="File path if the issue is tied to a specific file.",
    )
    line: int | None = Field(
        None,
        description="Line number if available.",
    )
    suggestion: str | None = Field(
        None,
        description="Optional fix or mitigation guidance.",
    )


class ReviewDelegateOutput(BaseModel):
    """Structured response for review delegate runs."""

    model_config = ConfigDict(extra="ignore")

    summary: str = Field(..., description="Overall review summary.")
    findings: list[ReviewFinding] = Field(
        default_factory=list,
        description="Specific findings with severity and optional location details.",
    )
    tests: list[str] = Field(
        default_factory=list,
        description="Missing or recommended tests.",
    )
