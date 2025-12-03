"""Planning sub-agent and tool wiring using pydantic-ai agent delegation.

This implements a dedicated planning agent that the main executor can call as a
tool. It aligns with the ACP agent-plan section:
https://agentclientprotocol.com/protocol/agent-plan

The planning agent may use only read-only tools (list/read/search) and must
never execute or mutate files. It is also prevented from recursive planning.

The planning delegate uses structured output (list of steps) to avoid text parsing.
"""

from __future__ import annotations

from typing import Any, List

from pydantic_ai import Agent as PydanticAgent  # type: ignore
from pydantic import BaseModel
import logging

from isaac.agent.brain.planner import parse_plan_from_text

PLANNING_SYSTEM_PROMPT = """
You are Isaac's planning delegate.
- Produce only a concise plan for the requested task as a list of short steps.
- Return 3-6 steps max.
- No execution, no code edits, and no extra narrative.
- Keep steps specific and outcome-focused so the executor can follow them.
"""


class PlanSteps(BaseModel):
    steps: List[str]


def build_planning_agent(model: Any, model_settings: Any = None) -> PydanticAgent:
    """Create a lightweight planning agent for delegation calls.

    This sub-agent is only responsible for planning and never executes tools.
    The plan format matches ACP agent-plan expectations so the outer agent can
    emit proper AgentPlanUpdate notifications.
    """

    return PydanticAgent(
        model,
        system_prompt=PLANNING_SYSTEM_PROMPT,
        model_settings=model_settings,
        toolsets=(),  # explicit empty toolsets; read-only tools are added separately
        output_type=PlanSteps,
    )


def make_planning_tool(planning_agent: PydanticAgent):
    """Return a coroutine tool that invokes the planning sub-agent."""

    async def tool_generate_plan(**kwargs: str | None) -> dict[str, Any]:
        # Accept flexible argument names to handle model/provider quirks.
        goal = kwargs.get("goal") or kwargs.get("plan") or kwargs.get("query") or ""
        context = kwargs.get("context") or kwargs.get("details") or None
        prompt = goal if not context else f"{goal}\n\nContext:\n{context}"
        logger = logging.getLogger("acp_server")
        logger.info("Planning delegate start prompt len=%s goal_preview=%s", len(prompt), goal[:80])
        try:
            result = await planning_agent.run(prompt)
            steps: list[str] = []
            plan_text = ""
            if result:
                if getattr(result, "output", None):
                    out = getattr(result, "output")
                    if isinstance(out, PlanSteps):
                        steps = out.steps
                    elif isinstance(out, list):
                        steps = [str(s) for s in out]
                # Fallback: parse any text parts if structured output failed.
                try:
                    response = getattr(result, "response", None)
                    parts = getattr(response, "parts", None) or []
                    texts = [getattr(p, "content", "") for p in parts if hasattr(p, "content")]
                    plan_text = "\n".join(str(t) for t in texts if str(t))
                    if not steps and plan_text:
                        parsed = parse_plan_from_text(plan_text)
                        if parsed and getattr(parsed, "entries", None):
                            steps = [e.content for e in parsed.entries]
                except Exception:  # pragma: no cover - best effort fallback
                    pass
            if not steps:
                steps = [
                    f"Clarify/confirm goal: {goal}",
                    "Draft necessary changes",
                    "Review, test, and deliver the result",
                ]
            logger.info(
                "Planning delegate parsed steps=%s text_preview=%s",
                len(steps),
                plan_text[:160].replace("\n", "\\n"),
            )
            content_text = "Plan:\n" + "\n".join(f"- {s}" for s in steps)
            logger.info(
                "Planning delegate done steps=%s preview=%s",
                len(steps),
                "; ".join(steps)[:120],
            )
            return {"plan_steps": steps, "content": content_text, "error": None}
        except Exception as exc:  # pragma: no cover - provider/runtime errors
            logger.warning("Planning delegate failed: %s", exc)
            fallback_steps = [
                f"Clarify/confirm goal: {goal}",
                "Draft necessary changes",
                "Review, test, and deliver the result",
            ]
            return {
                "plan_steps": fallback_steps,
                "content": "Plan:\n" + "\n".join(f"- {s}" for s in fallback_steps),
                "error": str(exc),
            }

    tool_generate_plan.__doc__ = (
        "Call the planning delegate to generate an ACP-compatible plan (agent-plan section)."
    )
    return tool_generate_plan
