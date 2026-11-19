"""Small ACP presentation bridge over public Harness capabilities."""

import logging
from dataclasses import dataclass, field, replace

from acp import schema
from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering, WrapModelRequestHandler, on_event
from pydantic_ai.exceptions import FallbackExceptionGroup, ModelAPIError, UnexpectedModelBehavior, UserError
from pydantic_ai.messages import CompactionPart, FunctionToolResultEvent, ModelResponse, TextPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.openai_codex import OpenAICodexModel
from pydantic_ai_harness.compaction import (
    ContextUsageMeasuredEvent,
    FallbackCompaction,
    SlidingWindowCompaction,
    SummarizingCompaction,
    compact_now,
    estimate_message_tokens,
)
from pydantic_ai_harness.experimental.acp import ACPSession, PydanticAIACPAgent
from pydantic_ai_harness.planning import PlanStore, TaskStatus, render_plan

from isaac.compaction import AutomaticCompaction
from isaac.config import PERMISSION_MODES, SessionSettings
from isaac.plan_mode import PLAN_COMMANDS
from isaac.session_store import SqliteSessionStore

logger = logging.getLogger(__name__)

COMMANDS = (
    schema.AvailableCommand(name="help", description="Show Isaac's agent commands."),
    schema.AvailableCommand(name="checklist", description="Show the implementation checklist."),
    schema.AvailableCommand(name="title", description="Show the saved session title."),
    schema.AvailableCommand(name="models", description="List the models configured for Isaac."),
    schema.AvailableCommand(
        name="model",
        description="Show the current model or select one from /models.",
        input=schema.AvailableCommandInput(root=schema.UnstructuredCommandInput(hint="optional model ID")),
    ),
    schema.AvailableCommand(
        name="permissions",
        description="Show or change the permission mode for this session.",
        input=schema.AvailableCommandInput(root=schema.UnstructuredCommandInput(hint="auto | read-only | full")),
    ),
    schema.AvailableCommand(
        name="compact",
        description="Compact conversation history. Add a focus to say what the summary should preserve.",
        input=schema.AvailableCommandInput(root=schema.UnstructuredCommandInput(hint="optional focus for the summary")),
    ),
)
AVAILABLE_COMMANDS = (*PLAN_COMMANDS, *COMMANDS)


@dataclass
class SessionUI(AbstractCapability[SessionSettings]):
    """Keep presentation session-local; Harness still owns turns, locks, and cancellation."""

    session: ACPSession
    plan: PlanStore
    store: SqliteSessionStore
    adapter: PydanticAIACPAgent[SessionSettings, str]
    last_entries: list[schema.PlanEntry] = field(default_factory=list, init=False)

    def get_default_conversation_id(self, run_id: str) -> str:
        # Give the first run a stable lookup key even if it never commits a turn.
        return self.session.session_id

    def get_ordering(self) -> CapabilityOrdering:
        # Local commands can be answered before automatic history compaction makes a request.
        return CapabilityOrdering(wraps=[AutomaticCompaction])

    async def advertise(self) -> None:
        await self.session.client.session_update(
            session_id=self.session.session_id,
            update=schema.AvailableCommandsUpdate(
                session_update="available_commands_update", available_commands=list(AVAILABLE_COMMANDS)
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
    async def tool_finished(self, _ctx: RunContext[SessionSettings], _event: FunctionToolResultEvent) -> None:
        # Publish after the tool completes, once per snapshot. This also catches pure reorders,
        # which do not emit per-item Harness plan events.
        await self.publish_plan()

    @on_event(ContextUsageMeasuredEvent)
    async def context_usage(self, _ctx: RunContext[SessionSettings], event: ContextUsageMeasuredEvent) -> None:
        if event.resolved:
            await self.session.client.session_update(
                session_id=self.session.session_id,
                update=schema.UsageUpdate(
                    session_update="usage_update", used=event.used_tokens, size=event.window_tokens
                ),
            )

    async def wrap_model_request(
        self,
        ctx: RunContext[SessionSettings],
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
        if arguments and command not in {"/compact", "/model", "/permissions"}:
            reply = f"Usage: {command}"
        elif command == "/help":
            reply = "\n".join(f"/{item.name}: {item.description}" for item in AVAILABLE_COMMANDS)
        elif command == "/checklist":
            await self.publish_plan(force=True)
            items = await self.plan.get_items()
            reply = render_plan(items) if items else "The checklist is empty."
        elif command == "/title":
            reply = await self.store.get_title(self.session.session_id) or "No title yet. Send a coding request first."
        elif command in {"/model", "/models"}:
            reply = await self.model_command(arguments, list_models=command == "/models")
        elif command == "/permissions":
            reply = permission_command(ctx.deps, arguments)
        else:
            reply = await self.compact(ctx, request_context, arguments.strip() or None)
        # A public short-circuit response still passes through Harness's streaming/history path.
        return ModelResponse(parts=[TextPart(reply)], finish_reason="stop")

    async def model_command(self, model_id: str, *, list_models: bool) -> str:
        session_id = self.session.session_id
        option = self.adapter.session_model_option(session_id)
        if option is None:
            return "No model choices configured. Set ISAAC_MODEL or pass --model when starting Isaac."
        models = [entry.value for entry in option.options if isinstance(entry, schema.SessionConfigSelectOption)]
        if list_models:
            return "\n".join(f"- {model}{' (current)' if model == option.current_value else ''}" for model in models)
        if not model_id or model_id == option.current_value:
            return f"Current model is {option.current_value}. Use /models to see all choices."
        if model_id not in models:
            return f"Unknown model {model_id!r}. Use /models to see all choices."
        updated = await self.adapter.set_config_option(config_id=option.id, session_id=session_id, value=model_id)
        assert updated is not None
        await self.session.client.session_update(
            session_id=session_id,
            update=schema.ConfigOptionUpdate(
                session_update="config_option_update", config_options=updated.config_options
            ),
        )
        return f"Model set to {model_id} for this session."

    async def compact(self, ctx: RunContext[SessionSettings], request: ModelRequestContext, focus: str | None) -> str:
        # Keep the current command request outside the summary so its reply remains paired with it.
        history, current = ctx.messages[:-1], ctx.messages[-1:]
        tokens = sum(estimate_message_tokens(history))
        if not tokens:
            return "Nothing to compact: the conversation is empty."
        if focus is None:
            native = await compact_native(ctx, replace(request, messages=history))
            if native is not None:
                # Core replays from the boundary on compatible providers. Keep the prefix so
                # a provider that cannot read the artifact still sees the preceding history.
                ctx.messages[:] = [*history, native, *current]
                request.messages = list(ctx.messages)
                return f"Compacted conversation history using native compaction on {request.model.model_name}."

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


def permission_command(settings: SessionSettings, mode: str) -> str:
    if not mode:
        return f"Permission mode is {settings.permissions}. Choose auto, read-only, or full with /permissions."
    if mode not in PERMISSION_MODES:
        return "Usage: /permissions [auto|read-only|full]"
    settings.permissions = mode
    return f"Permission mode set to {mode} for this session."


async def compact_native(ctx: RunContext[SessionSettings], request: ModelRequestContext) -> ModelResponse | None:
    # The Codex subscription backend returns 404 for OpenAI's standalone /responses/compact.
    if isinstance(request.model, OpenAICodexModel):
        return None
    try:
        response = await request.model.compact_messages(request)
    except NotImplementedError:
        return None
    except (ModelAPIError, FallbackExceptionGroup, UnexpectedModelBehavior, UserError) as error:
        logger.debug("Native compaction unavailable for %s (%s)", request.model.model_name, type(error).__name__)
        return None

    ctx.usage.requests += 1
    ctx.usage.incr(response.usage)
    if any(
        isinstance(part, CompactionPart)
        and (part.has_content() or bool((part.provider_details or {}).get("encrypted_content")))
        and request.model.accepts_stateless_artifacts(response, part)
        for part in response.parts
    ):
        return response
    return None
