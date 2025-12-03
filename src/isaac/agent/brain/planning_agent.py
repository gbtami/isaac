"""Planning agent used via programmatic hand-off (no in-run delegation)."""

from __future__ import annotations

from typing import Any, List

from pydantic import BaseModel
from pydantic_ai import Agent as PydanticAgent  # type: ignore

PLANNING_SYSTEM_PROMPT = """
You are Isaac's dedicated planning agent.
- Produce only a concise plan as 3-6 short, outcome-focused steps.
- No execution, no code edits, and no extra narrative.
- Keep steps specific so the executor can follow them.
"""


class PlanSteps(BaseModel):
    steps: List[str]


def build_planning_agent(model: Any, model_settings: Any = None) -> PydanticAgent:
    """Create a lightweight planning agent for programmatic hand-off."""

    return PydanticAgent(
        model,
        system_prompt=PLANNING_SYSTEM_PROMPT,
        model_settings=model_settings,
        toolsets=(),
        output_type=PlanSteps,
    )
