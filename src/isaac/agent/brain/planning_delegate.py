"""Planning sub-agent and tool wiring using pydantic-ai agent delegation.

This implements a dedicated planning agent that the main executor can call as a
tool. It aligns with the ACP agent-plan section:
https://agentclientprotocol.com/protocol/agent-plan

The planning agent may use only read-only tools (list/read/search) and must
never execute or mutate files. It is also prevented from recursive planning.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai import Agent as PydanticAgent  # type: ignore

PLANNING_SYSTEM_PROMPT = """
You are Isaac's planning delegate.
- Produce only a concise plan for the requested task.
- Respond with a `Plan:` header followed by 3-6 bullet steps.
- No execution, no code edits, and no extra narrative.
- Keep steps specific and outcome-focused so the executor can follow them.
"""


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
    )


def make_planning_tool(planning_agent: PydanticAgent):
    """Return a coroutine tool that invokes the planning sub-agent."""

    async def tool_generate_plan(goal: str, context: str | None = None) -> dict[str, str | None]:
        prompt = goal if not context else f"{goal}\n\nContext:\n{context}"
        try:
            result = await planning_agent.run(prompt)
            output = getattr(result, "output", "") if result else ""
            if not output or "plan:" not in output.lower():
                output = (
                    "Plan:\n"
                    f"- Clarify/confirm goal: {goal}\n"
                    "- Draft the necessary changes\n"
                    "- Review, test, and deliver the result"
                )
            return {"content": output, "error": None}
        except Exception as exc:  # pragma: no cover - provider/runtime errors
            return {"content": None, "error": str(exc)}

    tool_generate_plan.__doc__ = (
        "Call the planning delegate to generate an ACP-compatible plan (agent-plan section)."
    )
    return tool_generate_plan
