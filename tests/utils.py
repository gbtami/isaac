from __future__ import annotations

import inspect
from unittest.mock import AsyncMock

from acp.agent.connection import AgentSideConnection
from acp import RequestPermissionResponse
from acp.schema import AllowedOutcome, AuthMethod
import httpx
from isaac.agent import ACPAgent
from isaac.agent.tools import register_tools
from pydantic_ai import Agent as PydanticAgent  # type: ignore
from pydantic_ai import DeferredToolRequests  # type: ignore
from pydantic_ai.models.test import TestModel  # type: ignore


def make_function_agent(
    conn: AgentSideConnection,
    *,
    auth_methods: list[AuthMethod | dict[str, object] | str] | None = None,
) -> ACPAgent:
    """Helper to build ACPAgent with a deterministic in-process model."""

    runner = PydanticAgent(TestModel(call_tools=[]), output_type=[str, DeferredToolRequests])
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
    return ACPAgent(
        conn,
        runner_factory=lambda *_args, **_kwargs: runner,
        auth_methods=auth_methods,
    )


def make_error_agent(conn: AgentSideConnection) -> ACPAgent:
    """Agent whose runner fails to simulate provider errors."""

    class ErrorRunner:
        async def run_stream_events(  # pragma: no cover - simple stub
            self, prompt: str, *, message_history=None
        ):
            raise RuntimeError("rate limited")

    return ACPAgent(conn, runner_factory=lambda *_args, **_kwargs: ErrorRunner())


def make_timeout_agent(conn: AgentSideConnection) -> ACPAgent:
    """Agent whose runner fails to simulate provider timeouts."""

    class TimeoutRunner:
        async def run_stream_events(  # pragma: no cover - simple stub
            self, prompt: str, *, message_history=None
        ):
            request = httpx.Request("GET", "https://example.com")
            raise httpx.ReadTimeout("Request timed out.", request=request)

    return ACPAgent(conn, runner_factory=lambda *_args, **_kwargs: TimeoutRunner())
