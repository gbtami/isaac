"""Delegate agent registry and shared execution helpers."""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import json
import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from pydantic import BaseModel
from pydantic_ai import Agent as PydanticAgent  # type: ignore
from pydantic_ai import DeferredToolRequests, FunctionToolCallEvent, FunctionToolResultEvent  # type: ignore
from pydantic_ai.exceptions import ModelRetry  # type: ignore
from pydantic_ai.messages import RetryPromptPart  # type: ignore
from pydantic_ai.usage import UsageLimits  # type: ignore

from acp.contrib.tool_calls import ToolCallTracker
from acp.helpers import session_notification, text_block, tool_content, tool_diff_content, update_agent_thought
from isaac.agent.ai_types import AgentRunner, SessionToolDeps
from isaac.agent import models as model_registry
from isaac.agent.brain.instrumentation import base_run_metadata
from isaac.agent.brain.prompt import SYSTEM_PROMPT
from isaac.agent.brain.tool_args import coerce_tool_args
from isaac.agent.brain.tool_events import tool_kind
from isaac.agent.capabilities import build_base_capabilities, build_event_stream_observer_capability
from isaac.agent.models import load_models_config, load_runtime_env, _build_provider_model
from isaac.agent.runner import stream_with_runner
from isaac.log_utils import log_context as log_ctx, log_event

DELEGATE_TOOL_TIMEOUT_S = 60.0
DELEGATE_TOOL_DEFAULT_DEPTH = 2
DELEGATE_TOOL_SUMMARY_MAX_CHARS = 1200
DELEGATE_REQUEST_LIMIT = 50
DELEGATE_TOOL_CALLS_LIMIT = 120

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
    mode_getter: Callable[[], str]
    cwd: Path | None = None
    additional_directories: tuple[Path, ...] = ()
    model_id: str = ""


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


def _build_delegate_agent(
    spec: DelegateToolSpec,
    *,
    mode_getter: Callable[[], str] | None = None,
    model_id: str | None = None,
) -> AgentRunner:
    """Create a delegate agent for the current session model."""

    load_runtime_env()
    config = load_models_config()
    model_id = model_id or model_registry.current_model_id()
    models_cfg = config.get("models", {})
    if model_id not in models_cfg:
        raise ValueError(f"Unknown model id: {model_id}")
    model_entry = models_cfg.get(model_id, {})

    model_obj, model_settings = _build_provider_model(model_id, model_entry)
    output_type = spec.output_type
    from isaac.agent.tools import TOOL_HANDLERS, build_isaac_tools_capability

    tool_names = _expand_tool_names(spec)
    unknown = [name for name in tool_names if name not in TOOL_HANDLERS]
    if unknown:
        raise ValueError(f"Unknown delegate tool(s): {', '.join(sorted(unknown))}")

    capabilities = build_base_capabilities(mode_getter or (lambda: "ask"))
    capabilities.append(
        build_isaac_tools_capability(
            tool_names=tool_names,
            capability_id=f"isaac-delegate-{spec.name}-tools",
        )
    )
    agent: AgentRunner = PydanticAgent(
        model_obj,
        output_type=[output_type or str, DeferredToolRequests],
        toolsets=(),
        system_prompt=spec.system_prompt or SYSTEM_PROMPT,
        instructions=spec.instructions,
        model_settings=model_settings,
        capabilities=capabilities,
        metadata=base_run_metadata(component=f"isaac.delegate.{spec.name}", model_id=model_id),
    )

    if output_type is not None and spec.summary_extractor is not None and spec.min_summary_chars > 0:

        @agent.output_validator
        async def _validate_structured_length(output: BaseModel) -> BaseModel:
            summary = spec.summary_extractor(output) or ""
            if len(summary.strip()) < spec.min_summary_chars:
                raise ModelRetry(spec.continuation_prompt or "Provide a fuller structured summary.")
            return output

    return agent


@dataclass
class _DelegateRunOutcome:
    """Container for delegate run results."""

    content: BaseModel | str | None
    raw_text: str | None
    error: str | None


def _summary_from_output(spec: DelegateToolSpec, output: BaseModel | str | object) -> str | None:
    """Return a summary string for carryover and brevity checks.

    The summary is stored in memory for the same delegate session only.
    """

    if spec.summary_extractor is not None:
        try:
            if isinstance(output, (BaseModel, str)):
                summary = spec.summary_extractor(output)
                if summary:
                    return summary
        except Exception:
            pass
    if isinstance(output, str):
        stripped = output.strip()
        if stripped.startswith("{"):
            with contextlib.suppress(Exception):
                parsed = json.loads(stripped)
                summary = parsed.get("summary") if isinstance(parsed, dict) else None
                if isinstance(summary, str) and summary.strip():
                    return summary
        return stripped or None
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



def _delegate_tool_event_capability(
    spec: DelegateToolSpec,
    *,
    delegate_run_id: str,
    session_id: str | None,
    send_update: Callable[[Any], Awaitable[None]] | None,
) -> Any | None:
    """Surface inner delegate tool calls to the parent ACP session.

    Delegate agents intentionally run with isolated history, but their tool
    activity is still important UX and debugging context for long coding
    sessions. Expose it through Pydantic AI's run-scoped event-stream
    capability instead of coupling delegate tools to the ACP adapter directly.
    """

    if send_update is None or not session_id:
        return None

    trackers: dict[str, ToolCallTracker] = {}
    call_inputs: dict[str, dict[str, Any]] = {}

    def _prefixed_id(tool_call_id: str) -> str:
        base = tool_call_id or uuid.uuid4().hex
        return f"delegate:{spec.name}:{delegate_run_id}:{base}"

    async def _emit(event: Any) -> None:
        if isinstance(event, FunctionToolCallEvent):
            part = getattr(event, "part", None)
            tool_name = str(getattr(part, "tool_name", "") or "")
            raw_args = getattr(part, "args", None)
            original_id = str(getattr(event, "tool_call_id", "") or getattr(part, "tool_call_id", "") or "")
            prefixed_id = _prefixed_id(original_id)
            args = coerce_tool_args(raw_args)
            call_inputs[prefixed_id] = args
            tracker = ToolCallTracker(id_factory=lambda: prefixed_id)
            trackers[prefixed_id] = tracker
            start = tracker.start(
                external_id=prefixed_id,
                title=f"{spec.name}: {tool_name or 'tool'}",
                kind=tool_kind(tool_name),
                status="in_progress",
                raw_input={
                    "delegate_tool": spec.name,
                    "delegate_run_id": delegate_run_id,
                    "tool": tool_name,
                    **args,
                },
            )
            await send_update(session_notification(session_id, start))
            return

        if isinstance(event, FunctionToolResultEvent):
            result_part = getattr(event, "result", None) or getattr(event, "part", None)
            tool_name = str(getattr(result_part, "tool_name", "") or "")
            original_id = str(getattr(event, "tool_call_id", "") or getattr(result_part, "tool_call_id", "") or "")
            prefixed_id = _prefixed_id(original_id)
            tracker = trackers.pop(prefixed_id, None)
            if tracker is None:
                tracker = ToolCallTracker(id_factory=lambda: prefixed_id)
                start = tracker.start(
                    external_id=prefixed_id,
                    title=f"{spec.name}: {tool_name or 'tool'}",
                    kind=tool_kind(tool_name),
                    status="in_progress",
                    raw_input={
                        "delegate_tool": spec.name,
                        "delegate_run_id": delegate_run_id,
                        "tool": tool_name,
                        **call_inputs.get(prefixed_id, {}),
                    },
                )
                await send_update(session_notification(session_id, start))

            content = getattr(result_part, "content", None)
            raw_output: dict[str, Any] = content.copy() if isinstance(content, dict) else {"content": content}
            raw_output.setdefault("tool", tool_name)
            raw_output.setdefault("delegate_tool", spec.name)
            raw_output.setdefault("delegate_run_id", delegate_run_id)
            status = "completed"
            if isinstance(result_part, RetryPromptPart):
                raw_output["error"] = result_part.model_response()
                status = "failed"
            else:
                raw_output.setdefault("error", None)
                error_text = str(raw_output.get("error") or "").strip()
                returncode = raw_output.get("returncode")
                if error_text or (isinstance(returncode, int) and returncode != 0):
                    status = "failed"

            content_blocks: list[Any] = []
            new_text = raw_output.get("new_text")
            old_text = raw_output.get("old_text")
            if tool_name in {"edit_file", "apply_patch"} and isinstance(new_text, str):
                path = str(raw_output.get("path") or "")
                with contextlib.suppress(Exception):
                    content_blocks.append(tool_diff_content(path, new_text, old_text if isinstance(old_text, str) else None))
            summary = raw_output.get("error") or raw_output.get("content") or ""
            if not content_blocks and summary:
                content_blocks = [tool_content(text_block(str(summary)))]
            progress = tracker.progress(
                external_id=prefixed_id,
                status=status,
                raw_output=raw_output,
                content=content_blocks or None,
            )
            await send_update(session_notification(session_id, progress))

    return build_event_stream_observer_capability(_emit)


async def _run_delegate_once(
    spec: DelegateToolSpec,
    agent: AgentRunner,
    prompt: str,
    *,
    delegate_run_id: str,
    log_context: str | None,
    tool_call_id: str | None,
    session_id: str | None,
    send_update: Callable[[Any], Awaitable[None]] | None,
) -> _DelegateRunOutcome:
    """Run a delegate agent once and capture structured output if available.

    Delegate text output is returned only after completion; no text chunk
    streaming. Thinking chunks may still stream as thought updates.
    """

    structured: BaseModel | None = None
    cancel_event = asyncio.Event()
    thought_buffer: list[str] = []
    thought_last_sent = 0.0

    async def _emit_thought(text: str) -> None:
        if not send_update or not session_id:
            return
        thought = text.strip()
        if not thought:
            return
        with contextlib.suppress(Exception):
            await send_update(session_notification(session_id, update_agent_thought(text_block(thought))))

    async def _flush_thought(force: bool = False) -> None:
        nonlocal thought_last_sent
        if not thought_buffer:
            return
        now = time.monotonic()
        if not force and (now - thought_last_sent) < 0.5 and sum(len(chunk) for chunk in thought_buffer) < 200:
            return
        text = "".join(thought_buffer)
        thought_buffer.clear()
        thought_last_sent = now
        await _emit_thought(text)

    async def _drop_text(_: str) -> None:
        return None

    async def _on_thought(chunk: str) -> None:
        if not chunk:
            return
        thought_buffer.append(chunk)
        await _flush_thought()

    async def _capture_result(output: Any) -> None:
        nonlocal structured
        if spec.output_type is not None and isinstance(output, spec.output_type):
            structured = output

    delegate_ctx = get_delegate_tool_context()
    deps = None
    if delegate_ctx is not None and delegate_ctx.cwd is not None:
        deps = SessionToolDeps(
            session_id=delegate_ctx.session_id,
            cwd=delegate_ctx.cwd,
            additional_directories=delegate_ctx.additional_directories,
            mode=delegate_ctx.mode_getter(),
            model_id=delegate_ctx.model_id,
        )

    async def _request_tool_approval(call_id: str, tool_name: str, args: dict[str, Any]) -> bool:
        if tool_name != "run_command":
            return True
        if delegate_ctx is None:
            return True
        mode = delegate_ctx.mode_getter()
        if mode != "ask":
            return True
        command = str(args.get("command") or "")
        cwd = args.get("cwd")
        routed_call_id = f"delegate:{spec.name}:{delegate_run_id}:{call_id}"
        return await delegate_ctx.request_run_permission(
            delegate_ctx.session_id,
            routed_call_id,
            command,
            cwd if isinstance(cwd, str) or cwd is None else str(cwd),
        )

    delegate_tool_events = _delegate_tool_event_capability(
        spec,
        delegate_run_id=delegate_run_id,
        session_id=session_id,
        send_update=send_update,
    )
    run_capabilities = [delegate_tool_events] if delegate_tool_events is not None else None

    async def _run_stream() -> tuple[str | None, Any | None]:
        return await stream_with_runner(
            agent,
            prompt,
            _drop_text,
            _on_thought,
            cancel_event,
            on_result=_capture_result,
            log_context=log_context,
            request_tool_approval=_request_tool_approval,
            deps=deps,
            capabilities=run_capabilities,
            usage_limits=UsageLimits(
                request_limit=DELEGATE_REQUEST_LIMIT,
                tool_calls_limit=DELEGATE_TOOL_CALLS_LIMIT,
            ),
            metadata=base_run_metadata(
                component=f"isaac.delegate.run.{spec.name}",
                model_id=(delegate_ctx.model_id if delegate_ctx and delegate_ctx.model_id else model_registry.current_model_id()),
                extra={
                    "session_id": session_id or "",
                    "delegate_run_id": delegate_run_id,
                    "tool_call_id": tool_call_id or "",
                },
            ),
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

    await _flush_thought(force=True)

    if response is None:
        return _DelegateRunOutcome(content=None, raw_text=None, error="Delegate agent cancelled.")
    if response.startswith("Provider timeout:"):
        msg = response.removeprefix("Provider timeout:").strip()
        return _DelegateRunOutcome(content=None, raw_text=response, error=msg)
    if response.startswith("Provider error:"):
        msg = response.removeprefix("Provider error:").strip()
        return _DelegateRunOutcome(content=None, raw_text=response, error=msg)
    if response.lower().startswith("model output failed validation"):
        return _DelegateRunOutcome(content=None, raw_text=response, error=response)

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
    mode_getter = delegate_ctx.mode_getter if delegate_ctx else (lambda: "ask")
    model_id = delegate_ctx.model_id if delegate_ctx and delegate_ctx.model_id else None
    try:
        agent = _build_delegate_agent(spec, mode_getter=mode_getter, model_id=model_id)
        active_agent = agent
        prompt = _build_delegate_prompt(task, context, carryover_summary)

        outcome = await _run_delegate_once(
            spec,
            active_agent,
            prompt,
            delegate_run_id=delegate_run_id,
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

        summary = _summary_from_output(spec, output)
        output_payload = output
        if hasattr(output, "model_dump") and spec.name != "planner":
            try:
                output_payload = output.model_dump()  # type: ignore[call-arg]
            except Exception:
                output_payload = output
        if spec.output_type is None and _should_request_continuation(spec, summary):
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
                delegate_run_id=delegate_run_id,
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
        _delegate_depth.reset(token)
