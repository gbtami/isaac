from __future__ import annotations

from acp import AgentSideConnection
from isaac.agent import ACPAgent
from isaac.agent.brain.planning_delegate import build_planning_agent
from isaac.agent.runner import register_tools
from pydantic_ai import Agent as PydanticAgent  # type: ignore
from pydantic_ai.models.test import TestModel  # type: ignore


def make_function_agent(conn: AgentSideConnection) -> ACPAgent:
    """Helper to build ACPAgent with a deterministic in-process model."""

    runner = PydanticAgent(TestModel(call_tools=[]))
    planning_runner = build_planning_agent(TestModel(call_tools=[]))
    register_tools(runner, planning_agent=planning_runner)
    return ACPAgent(conn, ai_runner=runner)


def make_error_agent(conn: AgentSideConnection) -> ACPAgent:
    """Agent whose runner fails to simulate provider errors."""

    class ErrorRunner:
        async def run_stream_events(self, prompt: str):  # pragma: no cover - simple stub
            raise RuntimeError("rate limited")

    return ACPAgent(conn, ai_runner=ErrorRunner())
