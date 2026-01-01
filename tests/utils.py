from __future__ import annotations

import inspect
from unittest.mock import AsyncMock

from acp.agent.connection import AgentSideConnection
from acp import RequestPermissionResponse
from acp.schema import AllowedOutcome
from isaac.agent import ACPAgent
from isaac.agent.brain import session_ops
from isaac.agent.tools import register_tools
from pydantic_ai import Agent as PydanticAgent  # type: ignore
from pydantic_ai.models.test import TestModel  # type: ignore


def make_function_agent(conn: AgentSideConnection) -> ACPAgent:
    """Helper to build ACPAgent with a deterministic in-process model."""

    runner = PydanticAgent(TestModel(call_tools=[]))
    register_tools(runner)
    if not inspect.iscoroutinefunction(getattr(conn, "session_update", None)):
        conn.session_update = AsyncMock()

    # Default permission responder for tests (can be overridden per test).
    async def _default_perm(_: object) -> RequestPermissionResponse:
        return RequestPermissionResponse(outcome=AllowedOutcome(option_id="allow_once", outcome="selected"))

    current_perm = getattr(conn, "request_permission", None)
    if not inspect.iscoroutinefunction(current_perm):
        if isinstance(current_perm, AsyncMock) and current_perm.return_value is not None:
            pass
        else:
            conn.request_permission = _default_perm  # type: ignore[attr-defined]
    # Patch model builders to use deterministic test agents.
    session_ops.create_subagent_for_model = lambda *_args, **_kwargs: runner  # type: ignore[assignment]

    return ACPAgent(conn)


def make_error_agent(conn: AgentSideConnection) -> ACPAgent:
    """Agent whose runner fails to simulate provider errors."""

    class ErrorRunner:
        async def run_stream_events(self, prompt: str):  # pragma: no cover - simple stub
            raise RuntimeError("rate limited")

    session_ops.create_subagent_for_model = lambda *_args, **_kwargs: ErrorRunner()  # type: ignore[assignment]

    return ACPAgent(conn)
