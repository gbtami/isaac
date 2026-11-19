"""Read-only planning and the explicit handoff to implementation."""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from acp import schema
from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering, WrapModelRequestHandler, WrapRunHandler
from pydantic_ai.messages import ModelResponse, TextPart, UserContent
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import ToolDefinition
from pydantic_ai_harness.experimental.acp import ACPSession

from isaac.compaction import AutomaticCompaction
from isaac.config import SessionSettings
from isaac.session_store import SqliteSessionStore

PROPOSED_PLAN = "isaac/plan-proposed"
_PROPOSAL_HEADING = "## Proposed plan"
PLAN_COMMANDS = (
    schema.AvailableCommand(
        name="plan",
        description="Enter read-only Plan mode. Add a goal to start planning.",
        input=schema.AvailableCommandInput(root=schema.UnstructuredCommandInput(hint="optional goal")),
    ),
    schema.AvailableCommand(name="implement", description="Approve the proposed plan and start implementing it."),
    schema.AvailableCommand(name="cancel-plan", description="Leave Plan mode without starting implementation."),
)
_PLAN_TOOLS = frozenset({"read_file", "list_files", "grep", "ask_user_question"})


def parse_plan_command(prompt: str | Sequence[UserContent] | None) -> tuple[str, str]:
    parts = [prompt] if isinstance(prompt, str) else list(prompt or [])
    if not all(isinstance(part, str) for part in parts):
        return "", ""
    command, *arguments = "".join(part for part in parts if isinstance(part, str)).strip().split(maxsplit=1) or [""]
    return command, arguments[0] if arguments else ""


def planning_turn(ctx: RunContext[SessionSettings]) -> bool:
    """Select the restricted toolset before the run opens its tools."""
    command, arguments = parse_plan_command(ctx.prompt)
    return command in {"/plan", "/cancel-plan"} or (ctx.deps.planning and (command != "/implement" or bool(arguments)))


@dataclass
class PlanMode(AbstractCapability[SessionSettings]):
    session: ACPSession
    store: SqliteSessionStore
    command: str = ""
    arguments: str = ""
    planning: bool = False
    implementing: bool = False

    async def for_run(self, ctx: RunContext[SessionSettings]) -> "PlanMode":
        command, arguments = parse_plan_command(ctx.prompt)
        return replace(
            self,
            command=command,
            arguments=arguments,
            planning=planning_turn(ctx),
            implementing=ctx.deps.planning and command == "/implement" and not arguments,
        )

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(wraps=[AutomaticCompaction])

    def get_instructions(self) -> str | None:
        if self.implementing:
            return (
                "The user's /implement command approves implementation of the latest plan proposed in this "
                "conversation. Follow that plan and the agreed scope. If there is no concrete proposal, ask "
                "what to implement before taking action."
            )
        if self.planning:
            return (
                "You are in read-only Plan mode. Investigate the repository with the available read tools and "
                "ask focused questions when the user needs to decide something. Treat requests to implement as "
                "planning input while this mode is active. Treat any text after /plan as the user's goal. "
                "Propose or revise a concrete implementation plan with scope, decisions, and verification steps. "
                "When the proposal is ready for approval, write it as your final answer starting with "
                f"{_PROPOSAL_HEADING!r} on its own line. Reserve that heading for complete proposals. "
                "Tell the user they can run /implement to approve it and start work."
            )
        return None

    async def prepare_tools(
        self, ctx: RunContext[SessionSettings], tool_defs: list[ToolDefinition]
    ) -> list[ToolDefinition]:
        return [tool for tool in tool_defs if tool.name in _PLAN_TOOLS] if self.planning else tool_defs

    async def publish(self, planning: bool) -> None:
        await self.session.client.session_update(
            session_id=self.session.session_id,
            update=schema.CurrentModeUpdate(
                session_update="current_mode_update", current_mode_id="plan" if planning else "code"
            ),
        )

    async def _transition(self, settings: SessionSettings, planning: bool) -> None:
        await self.store.set_planning(self.session.session_id, planning)
        settings.planning = planning
        await self.publish(planning)

    async def wrap_run(self, ctx: RunContext[SessionSettings], *, handler: WrapRunHandler) -> AgentRunResult[Any]:
        if (self.command == "/plan" and not ctx.deps.planning) or (
            ctx.deps.planning and self.command in {"/implement", "/cancel-plan"} and not self.arguments
        ):
            transition = asyncio.create_task(self._transition(ctx.deps, self.command == "/plan"))
            try:
                await asyncio.shield(transition)
            except asyncio.CancelledError:
                await asyncio.gather(transition, return_exceptions=True)
                raise
        return await handler()

    async def after_run(self, ctx: RunContext[SessionSettings], *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
        capabilities = self.session.client_capabilities
        meta = capabilities.field_meta if capabilities else None
        if ctx.deps.planning and meta and meta.get(PROPOSED_PLAN) is True and isinstance(result.output, str):
            heading, _, body = result.output.strip().partition("\n")
            if heading.rstrip() == _PROPOSAL_HEADING and body.strip():
                await self.session.client.ext_notification(PROPOSED_PLAN, {"sessionId": self.session.session_id})
        return result

    async def wrap_model_request(
        self,
        ctx: RunContext[SessionSettings],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        if self.command == "/plan" and not self.arguments:
            reply = "Plan mode is on. Describe what to plan, then use /implement when ready or /cancel-plan to leave."
        elif self.command in {"/implement", "/cancel-plan"} and self.arguments:
            reply = f"Usage: {self.command}"
        elif self.command == "/implement" and not self.implementing:
            reply = "Plan mode is inactive. Use /plan to start planning."
        elif self.command == "/cancel-plan":
            reply = "Plan mode is off."
        else:
            return await handler(request_context)
        return ModelResponse(parts=[TextPart(reply)], finish_reason="stop")
