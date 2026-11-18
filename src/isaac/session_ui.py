"""Small ACP presentation bridge over public Harness capabilities."""

from dataclasses import dataclass, field

from acp import schema
from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability, WrapModelRequestHandler, on_event
from pydantic_ai.exceptions import FallbackExceptionGroup, ModelAPIError, UnexpectedModelBehavior
from pydantic_ai.messages import FunctionToolResultEvent, ModelResponse, TextPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai_harness.compaction import (
    FallbackCompaction,
    SlidingWindowCompaction,
    SummarizingCompaction,
    compact_now,
    estimate_message_tokens,
)
from pydantic_ai_harness.experimental.acp import ACPSession
from pydantic_ai_harness.planning import InMemoryPlanStore, TaskStatus, render_plan

COMMANDS = (
    schema.AvailableCommand(name="help", description="Show Isaac's agent commands."),
    schema.AvailableCommand(name="plan", description="Show the current session's plan."),
    schema.AvailableCommand(
        name="compact",
        description="Compact conversation history using Harness; optionally say what to preserve.",
        input=schema.AvailableCommandInput(root=schema.UnstructuredCommandInput(hint="optional focus for the summary")),
    ),
)


@dataclass
class SessionUI(AbstractCapability[None]):
    """Keep presentation session-local; Harness still owns turns, locks, and cancellation."""

    session: ACPSession
    plan: InMemoryPlanStore
    last_entries: list[schema.PlanEntry] = field(default_factory=list, init=False)

    async def advertise(self) -> None:
        await self.session.client.session_update(
            session_id=self.session.session_id,
            update=schema.AvailableCommandsUpdate(
                session_update="available_commands_update", available_commands=list(COMMANDS)
            ),
        )

    async def publish_plan(self, *, force: bool = False) -> None:
        entries = [
            schema.PlanEntry(
                content=item.content,
                priority="medium",
                status="completed"
                if item.status is TaskStatus.completed
                else "in_progress"
                if item.status is TaskStatus.in_progress
                else "pending",
            )
            for item in await self.plan.get_items()
            if item.status is not TaskStatus.cancelled
        ]
        if force or entries != self.last_entries:
            await self.session.client.session_update(
                session_id=self.session.session_id,
                update=schema.AgentPlanUpdate(session_update="plan", entries=entries),
            )
            self.last_entries = entries

    @on_event(FunctionToolResultEvent)
    async def tool_finished(self, _ctx: RunContext[None], _event: FunctionToolResultEvent) -> None:
        # Publish after the tool completes, once per snapshot. This also catches pure reorders,
        # which do not emit per-item Harness plan events.
        await self.publish_plan()

    async def wrap_model_request(
        self,
        ctx: RunContext[None],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        parts = [ctx.prompt] if isinstance(ctx.prompt, str) else list(ctx.prompt or [])
        if not parts or not all(isinstance(part, str) for part in parts):
            return await handler(request_context)
        text = "".join(part for part in parts if isinstance(part, str)).strip()
        command, *rest = text.split(maxsplit=1) or [""]
        arguments = rest[0] if rest else ""
        if "\n" in text or command not in {f"/{item.name}" for item in COMMANDS}:
            return await handler(request_context)
        if command != "/compact" and arguments:
            reply = f"Usage: {command}"
        elif command == "/help":
            reply = "\n".join(f"/{item.name}: {item.description}" for item in COMMANDS)
        elif command == "/plan":
            await self.publish_plan(force=True)
            items = await self.plan.get_items()
            reply = render_plan(items) if items else "No plan yet."
        else:
            reply = await self.compact(ctx, request_context, arguments.strip() or None)
        # A public short-circuit response still passes through Harness's streaming/history path.
        return ModelResponse(parts=[TextPart(reply)], finish_reason="stop")

    async def compact(self, ctx: RunContext[None], request: ModelRequestContext, focus: str | None) -> str:
        # Keep the current command request outside the summary so its reply remains paired with it.
        history, current = ctx.messages[:-1], ctx.messages[-1:]
        tokens = sum(estimate_message_tokens(history))
        if not tokens:
            return "Nothing to compact: the conversation is empty."
        tail = min(50_000, tokens // 2)
        strategy: FallbackCompaction[None] = FallbackCompaction(
            fallback_chain=[
                SummarizingCompaction(keep_tokens=tail),
                SlidingWindowCompaction(keep_tokens=tail),
            ],
            fallback_on=(ModelAPIError, FallbackExceptionGroup, UnexpectedModelBehavior),
        )
        compacted = await compact_now(
            strategy,
            history,
            model=request.model,
            focus=focus,
            usage=ctx.usage,
            conversation_id=ctx.conversation_id,
            tracer=ctx.tracer,
        )
        saved = tokens - sum(estimate_message_tokens(compacted))
        if saved <= 0:
            return "Nothing to compact: the history would not become smaller."
        ctx.messages[:] = [*compacted, *current]
        request.messages = list(ctx.messages)
        return f"Compacted {len(history)} messages to {len(compacted)}; approximately {saved:,} tokens saved."
