"""Delegate agent registry and shared execution helpers."""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, cast

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent as PydanticAgent  # type: ignore
from pydantic_ai.run import AgentRunResultEvent  # type: ignore

from acp.contrib.tool_calls import ToolCallTracker
from acp.helpers import session_notification, text_block, tool_content
from isaac.agent.ai_types import AgentRunner
from isaac.agent import models as model_registry
from isaac.agent.brain.prompt import SYSTEM_PROMPT
from isaac.agent.models import ENV_FILE, load_models_config, _build_provider_model
from isaac.agent.runner import stream_with_runner
from isaac.agent.tools.run_command import RunCommandContext, reset_run_command_context, set_run_command_context
from isaac.log_utils import log_context as log_ctx, log_event

DELEGATE_TOOL_TIMEOUT_S = 60.0
DELEGATE_TOOL_DEFAULT_DEPTH = 2
DELEGATE_TOOL_SUMMARY_MAX_CHARS = 1200

_delegate_depth = contextvars.ContextVar("delegate_tool_depth", default=0)
_delegate_context = contextvars.ContextVar("delegate_tool_context", default=None)

_DelegateHandler = Callable[..., Awaitable[dict[str, Any]]]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DelegateToolContext:
    """Context installed by the prompt handler for delegate tools.

    This lets delegate runs route permission requests (for commands) back to the
    parent ACP session without leaking full chat history. It also exposes a
    send_update hook so sub-agents can surface progress to the ACP client.
    """

    session_id: str
    request_run_permission: Callable[[str, str, str, str | None], Awaitable[bool]]
    send_update: Callable[[Any], Awaitable[None]]


def set_delegate_tool_context(ctx: DelegateToolContext) -> contextvars.Token[DelegateToolContext | None]:
    """Install the delegate tool context for the current prompt turn.

    This keeps delegate tools isolated while still letting them request
    permissions and emit ACP updates through the parent session.
    """

    return _delegate_context.set(ctx)


def reset_delegate_tool_context(token: contextvars.Token[DelegateToolContext | None]) -> None:
    """Reset the delegate tool context after the prompt finishes."""

    _delegate_context.reset(token)


def get_delegate_tool_context() -> DelegateToolContext | None:
    """Return the active delegate tool context, if any."""

    return _delegate_context.get()


@dataclass(frozen=True)
class DelegateToolSpec:
    """Configuration for a delegate agent tool.

    Delegate tools wrap dedicated sub-agents. The spec controls tool access,
    output expectations, timeouts, and optional follow-up behavior when output
    is too brief for multi-turn tasks.
    """

    name: str
    description: str
    instructions: str
    system_prompt: str | None
    tool_names: tuple[str, ...]
    output_type: type[BaseModel] | None = None
    timeout_s: float = DELEGATE_TOOL_TIMEOUT_S
    max_depth: int = DELEGATE_TOOL_DEFAULT_DEPTH
    log_context: str | None = None
    include_delegate_tools: bool = False
    summary_extractor: Callable[[BaseModel | str], str | None] | None = None
    min_summary_chars: int = 0
    continuation_prompt: str | None = None
    allow_unstructured_fallback: bool = True


DELEGATE_TOOL_HANDLERS: dict[str, _DelegateHandler] = {}
DELEGATE_TOOL_ARG_MODELS: dict[str, type[BaseModel]] = {}
DELEGATE_TOOL_DESCRIPTIONS: dict[str, str] = {}
DELEGATE_TOOL_TIMEOUTS: dict[str, float] = {}


@dataclass
class DelegateSession:
    """Minimal state retained for multi-turn delegate work.

    We keep only a short summary for carryover to avoid full history sharing.
    """

    session_id: str
    tool_name: str
    runs: int = 0
    last_summary: str | None = None


DELEGATE_SESSIONS: dict[str, DelegateSession] = {}


def register_delegate_tool(
    spec: DelegateToolSpec,
    *,
    handler: _DelegateHandler,
    arg_model: type[BaseModel],
) -> None:
    """Register a delegate tool definition.

    Delegate tools register their handler and argument model so discovery and
    tool registration can stay data-driven.
    """
    DELEGATE_TOOL_HANDLERS[spec.name] = handler
    DELEGATE_TOOL_ARG_MODELS[spec.name] = arg_model
    DELEGATE_TOOL_DESCRIPTIONS[spec.name] = spec.description
    DELEGATE_TOOL_TIMEOUTS[spec.name] = spec.timeout_s


def _delegate_session_key(spec: DelegateToolSpec, session_id: str) -> str:
    """Return a stable key for indexing delegate sessions by tool and id."""

    return f"{spec.name}:{session_id}"


def _get_delegate_session(spec: DelegateToolSpec, session_id: str) -> DelegateSession:
    """Return or create the session record for the given delegate tool."""

    key = _delegate_session_key(spec, session_id)
    session = DELEGATE_SESSIONS.get(key)
    if session is None:
        session = DelegateSession(session_id=session_id, tool_name=spec.name)
        DELEGATE_SESSIONS[key] = session
    return session


def _truncate_summary(text: str) -> str:
    """Clamp delegate summaries so carryover context stays lightweight."""

    trimmed = text.strip()
    if len(trimmed) <= DELEGATE_TOOL_SUMMARY_MAX_CHARS:
        return trimmed
    return f"{trimmed[:DELEGATE_TOOL_SUMMARY_MAX_CHARS].rstrip()}..."


def _build_delegate_prompt(task: str, context: str | None, carryover_summary: str | None) -> str:
    """Build the delegate prompt without leaking parent history.

    Only the explicit context string (if provided) and an optional carryover
    summary from the same delegate session are included. No other history is
    injected.
    """

    task_text = task.strip()
    parts: list[str] = []
    if context:
        parts.append(context.strip())
    if carryover_summary:
        parts.append(f"Previous delegate summary:\n{carryover_summary.strip()}")
    if parts:
        joined = "\n\n".join(parts)
        return f"{joined}\n\nTask: {task_text}"
    return task_text


def _expand_tool_names(spec: DelegateToolSpec) -> tuple[str, ...]:
    """Expand tool names to include other delegate tools when allowed."""
    if not spec.include_delegate_tools:
        return spec.tool_names
    tool_names = list(spec.tool_names)
    for name in DELEGATE_TOOL_HANDLERS.keys():
        if name == spec.name:
            continue
        if name not in tool_names:
            tool_names.append(name)
    return tuple(tool_names)


_OUTPUT_TYPE_UNSET = object()


def _build_delegate_agent(
    spec: DelegateToolSpec, *, output_type_override: type[BaseModel] | None | object = _OUTPUT_TYPE_UNSET
) -> AgentRunner:
    """Create a delegate agent with optional output-type override.

    The override is used for fallback retries when structured output validation
    fails. In that case we re-run without output_type to preserve usability.
    """

    load_dotenv(ENV_FILE, override=False)
    load_dotenv()
    config = load_models_config()
    model_id = model_registry.current_model_id()
    models_cfg = config.get("models", {})
    if model_id not in models_cfg:
        raise ValueError(f"Unknown model id: {model_id}")
    model_entry = models_cfg.get(model_id, {})

    model_obj, model_settings = _build_provider_model(model_id, model_entry)
    if output_type_override is _OUTPUT_TYPE_UNSET:
        output_type = spec.output_type
    else:
        output_type = cast(type[BaseModel] | None, output_type_override)
    agent: AgentRunner = PydanticAgent(
        model_obj,
        toolsets=(),
        system_prompt=spec.system_prompt or SYSTEM_PROMPT,
        instructions=spec.instructions,
        model_settings=model_settings,
        output_type=output_type,
    )

    from isaac.agent.tools import TOOL_HANDLERS, register_tools

    tool_names = _expand_tool_names(spec)
    unknown = [name for name in tool_names if name not in TOOL_HANDLERS]
    if unknown:
        raise ValueError(f"Unknown delegate tool(s): {', '.join(sorted(unknown))}")
    register_tools(agent, tool_names=tool_names)
    return agent


@dataclass
class _DelegateRunOutcome:
    """Container for delegate run results and validation status."""

    content: BaseModel | str | None
    raw_text: str | None
    error: str | None
    validation_failed: bool = False


def _summary_from_output(spec: DelegateToolSpec, output: BaseModel | str | object) -> str | None:
    """Return a summary string for carryover and brevity checks.

    The summary is stored in memory for the same delegate session only.
    """

    if spec.summary_extractor is not None:
        try:
            if isinstance(output, (BaseModel, str)):
                return spec.summary_extractor(output)
            return None
        except Exception:
            return None
    if isinstance(output, str):
        return output
    return None


def _should_request_continuation(spec: DelegateToolSpec, summary: str | None) -> bool:
    """Decide whether to ask the delegate for a more detailed response.

    This lets structured delegates retry with a follow-up prompt if their first
    response is too brief for the task at hand.
    """

    if not spec.continuation_prompt or spec.min_summary_chars <= 0:
        return False
    if summary is None:
        return True
    return len(summary.strip()) < spec.min_summary_chars


def _extract_json_candidate(text: str) -> str | None:
    """Best-effort extraction of a JSON object from a text blob."""

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or start >= end:
        return None
    return text[start : end + 1]


def _parse_structured_output(spec: DelegateToolSpec, text: str) -> BaseModel | None:
    """Try to parse a structured output payload using the spec's output_type."""

    output_type = spec.output_type
    if output_type is None:
        return None
    if not hasattr(output_type, "model_validate_json"):
        return None
    candidate = text.strip()
    try:
        return output_type.model_validate_json(candidate)
    except Exception:
        pass
    extracted = _extract_json_candidate(candidate)
    if extracted:
        try:
            return output_type.model_validate_json(extracted)
        except Exception:
            return None
    return None


def _build_permission_context(
    spec: DelegateToolSpec,
    delegate_run_id: str,
) -> tuple[contextvars.Token[RunCommandContext | None] | None, Callable[[], None]]:
    """Install a run_command permission bridge for delegate tool runs.

    Delegate tools can call run_command, but permissions must be routed through
    the parent ACP session so the user sees a single permission workflow.
    """

    ctx = get_delegate_tool_context()
    if ctx is None:
        return None, lambda: None
    if "run_command" not in _expand_tool_names(spec):
        return None, lambda: None

    counter = 0

    async def _request(command: str, cwd: str | None = None) -> bool:
        nonlocal counter
        counter += 1
        tool_call_id = f"delegate:{spec.name}:{delegate_run_id}:{counter}"
        return await ctx.request_run_permission(ctx.session_id, tool_call_id, command, cwd)

    token = set_run_command_context(RunCommandContext(request_permission=_request))

    def _reset() -> None:
        reset_run_command_context(token)

    return token, _reset


async def _run_delegate_once(
    spec: DelegateToolSpec,
    agent: AgentRunner,
    prompt: str,
    *,
    log_context: str | None,
    tool_call_id: str | None,
    session_id: str | None,
    send_update: Callable[[Any], Awaitable[None]] | None,
) -> _DelegateRunOutcome:
    """Run a delegate agent once and capture structured output if available.

    We stream incremental text back to the ACP client as tool progress updates,
    but only surface response text (never internal reasoning).
    """

    structured: BaseModel | None = None
    cancel_event = asyncio.Event()

    tracker: ToolCallTracker | None = None
    if tool_call_id and session_id and send_update:
        tracker = ToolCallTracker(id_factory=lambda: tool_call_id)
        # Seed tracker state so progress updates can reuse the existing ACP tool call id
        # without emitting a duplicate ToolCallStart update.
        tracker.start(
            external_id=tool_call_id,
            title=spec.name,
            status="in_progress",
            raw_input={"tool": spec.name},
        )

    buffer: list[str] = []
    last_sent = 0.0

    async def _emit_progress(text: str) -> None:
        if not tracker or not send_update or not session_id:
            return
        raw_output = {
            "tool": spec.name,
            "content": text,
            "delegate_tool": spec.name,
        }
        progress = tracker.progress(
            external_id=tool_call_id,
            status="in_progress",
            raw_output=raw_output,
            content=[tool_content(text_block(text))],
        )
        with contextlib.suppress(Exception):
            await send_update(session_notification(session_id, progress))

    async def _flush(force: bool = False) -> None:
        nonlocal last_sent
        if not buffer:
            return
        now = time.monotonic()
        if not force and (now - last_sent) < 0.5 and sum(len(chunk) for chunk in buffer) < 200:
            return
        text = "".join(buffer)
        buffer.clear()
        last_sent = now
        await _emit_progress(text)

    async def _on_text(chunk: str) -> None:
        if not chunk:
            return
        buffer.append(chunk)
        await _flush()

    async def _capture(event: Any) -> bool:
        nonlocal structured
        if spec.output_type is None:
            return False
        if isinstance(event, AgentRunResultEvent):
            output = getattr(event.result, "output", None)
            if isinstance(output, spec.output_type):
                structured = output
        return False

    async def _run_stream() -> tuple[str | None, Any | None]:
        return await stream_with_runner(
            agent,
            prompt,
            _on_text,
            None,
            cancel_event,
            on_event=_capture,
            log_context=log_context,
        )

    try:
        if spec.timeout_s > 0:
            response, _ = await asyncio.wait_for(_run_stream(), timeout=spec.timeout_s)
        else:
            response, _ = await _run_stream()
    except asyncio.TimeoutError:
        cancel_event.set()
        return _DelegateRunOutcome(
            content=None,
            raw_text=None,
            error=f"Delegate tool timed out after {spec.timeout_s}s.",
        )

    await _flush(force=True)

    if response is None:
        return _DelegateRunOutcome(content=None, raw_text=None, error="Delegate agent cancelled.")
    if response.startswith("Provider error:"):
        msg = response.removeprefix("Provider error:").strip()
        return _DelegateRunOutcome(content=None, raw_text=response, error=msg)
    if response.lower().startswith("model output failed validation"):
        return _DelegateRunOutcome(
            content=None,
            raw_text=response,
            error=response,
            validation_failed=True,
        )

    output = structured or response
    return _DelegateRunOutcome(content=output, raw_text=response, error=None)


async def run_delegate_tool(
    spec: DelegateToolSpec,
    *,
    task: str,
    context: str | None = None,
    session_id: str | None = None,
    carryover: bool = False,
    tool_call_id: str | None = None,
) -> dict[str, Any]:
    """Run a delegate agent tool with depth guarding.

    Delegate runs start with a fresh context. To continue multi-turn work,
    supply a `session_id` and set `carryover=True` to include the prior
    delegate summary in the prompt.

    Returns content/error plus delegate metadata (session/run ids) so callers
    can stitch multi-turn workflows together explicitly.
    """

    depth = _delegate_depth.get()
    if depth >= spec.max_depth:
        with log_ctx(delegate_tool=spec.name):
            log_event(
                logger,
                "delegate.run.depth_exceeded",
                level=logging.WARNING,
                max_depth=spec.max_depth,
                depth=depth,
            )
        return {
            "content": "",
            "error": f"Delegate tool depth exceeded (max {spec.max_depth}).",
            "delegate_tool": spec.name,
        }

    delegate_session_id = session_id or uuid.uuid4().hex
    delegate_run_id = uuid.uuid4().hex
    session = _get_delegate_session(spec, delegate_session_id)
    carryover_summary = session.last_summary if carryover else None
    with log_ctx(
        delegate_tool=spec.name,
        delegate_session_id=delegate_session_id,
        delegate_run_id=delegate_run_id,
    ):
        log_event(
            logger,
            "delegate.run.start",
            carryover=carryover,
            task_preview=task[:160].replace("\n", "\\n"),
        )

    token = _delegate_depth.set(depth + 1)
    delegate_ctx = get_delegate_tool_context()
    send_update = delegate_ctx.send_update if delegate_ctx else None
    parent_session_id = delegate_ctx.session_id if delegate_ctx else None

    def _reset_permission() -> None:
        return None

    reset_permission = _reset_permission
    try:
        agent = _build_delegate_agent(spec)
        active_agent = agent
        prompt = _build_delegate_prompt(task, context, carryover_summary)
        _, reset_permission = _build_permission_context(spec, delegate_run_id)

        outcome = await _run_delegate_once(
            spec,
            active_agent,
            prompt,
            log_context=spec.log_context,
            tool_call_id=tool_call_id,
            session_id=parent_session_id,
            send_update=send_update,
        )
        if outcome.validation_failed and spec.allow_unstructured_fallback:
            fallback_agent = _build_delegate_agent(spec, output_type_override=None)
            active_agent = fallback_agent
            outcome = await _run_delegate_once(
                spec,
                active_agent,
                prompt,
                log_context=spec.log_context,
                tool_call_id=tool_call_id,
                session_id=parent_session_id,
                send_update=send_update,
            )

        if outcome.error:
            with log_ctx(
                delegate_tool=spec.name,
                delegate_session_id=delegate_session_id,
                delegate_run_id=delegate_run_id,
            ):
                log_event(logger, "delegate.run.error", level=logging.WARNING, error=outcome.error)
            return {
                "content": "",
                "error": outcome.error,
                "delegate_tool": spec.name,
                "delegate_session_id": delegate_session_id,
                "delegate_run_id": delegate_run_id,
            }

        output = outcome.content
        if isinstance(output, str):
            parsed = _parse_structured_output(spec, output)
            if parsed is not None:
                output = parsed

        summary = _summary_from_output(spec, output)
        output_payload = output
        if hasattr(output, "model_dump") and spec.name != "planner":
            try:
                output_payload = output.model_dump()  # type: ignore[call-arg]
            except Exception:
                output_payload = output
        if _should_request_continuation(spec, summary):
            context_parts = []
            if context:
                context_parts.append(context.strip())
            if spec.continuation_prompt:
                context_parts.append(spec.continuation_prompt.strip())
            if isinstance(output, str) and output.strip():
                context_parts.append(f"Previous output:\n{output.strip()}")
            followup_context = "\n\n".join(context_parts) if context_parts else None
            followup_prompt = _build_delegate_prompt(task, followup_context, carryover_summary)
            outcome = await _run_delegate_once(
                spec,
                active_agent,
                followup_prompt,
                log_context=spec.log_context,
                tool_call_id=tool_call_id,
                session_id=parent_session_id,
                send_update=send_update,
            )
            if outcome.error:
                return {
                    "content": "",
                    "error": outcome.error,
                    "delegate_tool": spec.name,
                    "delegate_session_id": delegate_session_id,
                    "delegate_run_id": delegate_run_id,
                }
            output = outcome.content
            if isinstance(output, str):
                parsed = _parse_structured_output(spec, output)
                if parsed is not None:
                    output = parsed
            summary = _summary_from_output(spec, output)
            output_payload = output
            if hasattr(output, "model_dump") and spec.name != "planner":
                try:
                    output_payload = output.model_dump()  # type: ignore[call-arg]
                except Exception:
                    output_payload = output

        session.runs += 1
        if summary:
            session.last_summary = _truncate_summary(summary)
        with log_ctx(
            delegate_tool=spec.name,
            delegate_session_id=delegate_session_id,
            delegate_run_id=delegate_run_id,
        ):
            log_event(
                logger,
                "delegate.run.complete",
                summary_preview=(summary or "")[:160].replace("\n", "\\n"),
            )

        return {
            "content": output_payload,
            "error": None,
            "delegate_tool": spec.name,
            "delegate_session_id": delegate_session_id,
            "delegate_run_id": delegate_run_id,
        }
    except Exception as exc:
        with log_ctx(
            delegate_tool=spec.name,
            delegate_session_id=delegate_session_id,
            delegate_run_id=delegate_run_id,
        ):
            log_event(logger, "delegate.run.error", level=logging.WARNING, error=str(exc))
        return {
            "content": "",
            "error": str(exc),
            "delegate_tool": spec.name,
            "delegate_session_id": delegate_session_id,
            "delegate_run_id": delegate_run_id,
        }
    finally:
        reset_permission()
        _delegate_depth.reset(token)
