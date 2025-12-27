"""Delegate agent registry and shared execution helpers."""

from __future__ import annotations

import asyncio
import contextvars
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from dotenv import load_dotenv
from pydantic_ai import Agent as PydanticAgent  # type: ignore
from pydantic_ai.run import AgentRunResultEvent  # type: ignore

from isaac.agent import models as model_registry
from isaac.agent.brain.prompt import SYSTEM_PROMPT
from isaac.agent.models import ENV_FILE, load_models_config, _build_provider_model
from isaac.agent.runner import stream_with_runner
from isaac.agent.tools.run_command import RunCommandContext, reset_run_command_context, set_run_command_context

DELEGATE_TOOL_TIMEOUT_S = 60.0
DELEGATE_TOOL_DEFAULT_DEPTH = 2
DELEGATE_TOOL_SUMMARY_MAX_CHARS = 1200

_delegate_depth = contextvars.ContextVar("delegate_tool_depth", default=0)
_delegate_context = contextvars.ContextVar("delegate_tool_context", default=None)

_DelegateHandler = Callable[..., Any]

logger = logging.getLogger("isaac.delegate")


@dataclass(frozen=True)
class DelegateToolContext:
    """Context installed by the prompt handler for delegate tools.

    This lets delegate runs route permission requests (for commands) back to the
    parent ACP session without leaking full chat history.
    """

    session_id: str
    request_run_permission: Callable[[str, str, str, str | None], Awaitable[bool]]


def set_delegate_tool_context(ctx: DelegateToolContext) -> contextvars.Token[DelegateToolContext | None]:
    """Install the delegate tool context for the current prompt turn."""

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
    output_type: Any | None = None
    timeout_s: float = DELEGATE_TOOL_TIMEOUT_S
    max_depth: int = DELEGATE_TOOL_DEFAULT_DEPTH
    log_context: str | None = None
    include_delegate_tools: bool = False
    summary_extractor: Callable[[Any], str | None] | None = None
    min_summary_chars: int = 0
    continuation_prompt: str | None = None
    allow_unstructured_fallback: bool = True


DELEGATE_TOOL_HANDLERS: dict[str, _DelegateHandler] = {}
DELEGATE_TOOL_ARG_MODELS: dict[str, type[Any]] = {}
DELEGATE_TOOL_DESCRIPTIONS: dict[str, str] = {}
DELEGATE_TOOL_TIMEOUTS: dict[str, float] = {}


@dataclass
class DelegateSession:
    """Minimal state retained for multi-turn delegate work."""

    session_id: str
    tool_name: str
    runs: int = 0
    last_summary: str | None = None


DELEGATE_SESSIONS: dict[str, DelegateSession] = {}


def register_delegate_tool(
    spec: DelegateToolSpec,
    *,
    handler: _DelegateHandler,
    arg_model: type[Any],
) -> None:
    """Register a delegate tool definition."""
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


def _build_delegate_agent(spec: DelegateToolSpec, *, output_type_override: Any = _OUTPUT_TYPE_UNSET) -> Any:
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
    output_type = spec.output_type if output_type_override is _OUTPUT_TYPE_UNSET else output_type_override
    agent = PydanticAgent(
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

    content: Any | None
    raw_text: str | None
    error: str | None
    validation_failed: bool = False


def _summary_from_output(spec: DelegateToolSpec, output: Any) -> str | None:
    """Return a summary string for carryover and brevity checks."""

    if spec.summary_extractor is not None:
        try:
            return spec.summary_extractor(output)
        except Exception:
            return None
    if isinstance(output, str):
        return output
    return None


def _should_request_continuation(spec: DelegateToolSpec, summary: str | None) -> bool:
    """Decide whether to ask the delegate for a more detailed response."""

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


def _parse_structured_output(spec: DelegateToolSpec, text: str) -> Any | None:
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
    """Install a run_command permission bridge for delegate tool runs."""

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
    agent: Any,
    prompt: str,
    *,
    log_context: str | None,
) -> _DelegateRunOutcome:
    """Run a delegate agent once and capture structured output if available."""

    structured: Any | None = None
    cancel_event = asyncio.Event()

    async def _noop(_: str) -> None:
        return None

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
            _noop,
            _noop,
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
        return {
            "content": "",
            "error": f"Delegate tool depth exceeded (max {spec.max_depth}).",
            "delegate_tool": spec.name,
        }

    delegate_session_id = session_id or uuid.uuid4().hex
    delegate_run_id = uuid.uuid4().hex
    session = _get_delegate_session(spec, delegate_session_id)
    carryover_summary = session.last_summary if carryover else None
    logger.info(
        "Delegate run start tool=%s session=%s run=%s carryover=%s",
        spec.name,
        delegate_session_id,
        delegate_run_id,
        carryover,
    )

    token = _delegate_depth.set(depth + 1)

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
        )
        if outcome.validation_failed and spec.allow_unstructured_fallback:
            fallback_agent = _build_delegate_agent(spec, output_type_override=None)
            active_agent = fallback_agent
            outcome = await _run_delegate_once(
                spec,
                active_agent,
                prompt,
                log_context=spec.log_context,
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

        return {
            "content": output_payload,
            "error": None,
            "delegate_tool": spec.name,
            "delegate_session_id": delegate_session_id,
            "delegate_run_id": delegate_run_id,
        }
    except Exception as exc:
        logger.warning(
            "Delegate run error tool=%s session=%s run=%s error=%s",
            spec.name,
            delegate_session_id,
            delegate_run_id,
            exc,
        )
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
