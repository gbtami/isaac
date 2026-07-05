from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
import inspect
from unittest.mock import AsyncMock

import httpx
from acp import RequestPermissionResponse
from acp.agent.connection import AgentSideConnection
from acp.schema import AllowedOutcome, AuthMethodAgent, EnvVarAuthMethod, TerminalAuthMethod
from isaac.agent import ACPAgent
from isaac.agent.tools import build_isaac_tools_capability
from pydantic_ai import Agent as PydanticAgent  # type: ignore
from pydantic_ai import DeferredToolRequests  # type: ignore
from pydantic_ai.models.test import TestModel  # type: ignore

from typing import Any

AuthMethod = AuthMethodAgent | EnvVarAuthMethod | TerminalAuthMethod


async def notify_process_event_stream_capabilities(events: Sequence[Any], capabilities: Sequence[Any] | None) -> None:
    """Let fake runners notify Pydantic AI ProcessEventStream observers.

    Real Pydantic AI agents execute these capabilities internally. A few tests use
    tiny runner fakes that just yield events, so they explicitly notify observer
    capabilities before replaying the same events downstream.
    """

    if not capabilities:
        return

    async def _events() -> AsyncIterator[Any]:
        for event in events:
            yield event

    for capability in capabilities:
        if type(capability).__name__ != "ProcessEventStream":
            continue
        handler = getattr(capability, "handler", None)
        if handler is None:
            continue
        result = handler(None, _events())
        if inspect.isasyncgen(result):
            async for _ in result:
                pass
        elif inspect.isawaitable(result):
            await result


def make_function_agent(
    conn: AgentSideConnection,
    *,
    auth_methods: list[AuthMethod | dict[str, object] | str] | None = None,
) -> ACPAgent:
    """Helper to build ACPAgent with a deterministic in-process model."""

    runner = PydanticAgent(
        TestModel(call_tools=[]),
        output_type=[str, DeferredToolRequests],
        toolsets=(),
        capabilities=[build_isaac_tools_capability()],
    )
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
            self, prompt: str, *, message_history=None, **_: object
        ):
            raise RuntimeError("rate limited")

    return ACPAgent(conn, runner_factory=lambda *_args, **_kwargs: ErrorRunner())


def make_timeout_agent(conn: AgentSideConnection) -> ACPAgent:
    """Agent whose runner fails to simulate provider timeouts."""

    class TimeoutRunner:
        async def run_stream_events(  # pragma: no cover - simple stub
            self, prompt: str, *, message_history=None, **_: object
        ):
            request = httpx.Request("GET", "https://example.com")
            raise httpx.ReadTimeout("Request timed out.", request=request)

    return ACPAgent(conn, runner_factory=lambda *_args, **_kwargs: TimeoutRunner())
