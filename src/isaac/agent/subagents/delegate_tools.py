"""Delegate agent registry and shared execution helpers."""

from __future__ import annotations

import asyncio
import contextvars
from dataclasses import dataclass
from typing import Any, Callable

from dotenv import load_dotenv
from pydantic_ai import Agent as PydanticAgent  # type: ignore
from pydantic_ai.run import AgentRunResultEvent  # type: ignore

from isaac.agent import models as model_registry
from isaac.agent.brain.prompt import SYSTEM_PROMPT
from isaac.agent.models import ENV_FILE, load_models_config, _build_provider_model
from isaac.agent.runner import stream_with_runner

DELEGATE_TOOL_TIMEOUT_S = 30.0
DELEGATE_TOOL_DEFAULT_DEPTH = 2

_delegate_depth = contextvars.ContextVar("delegate_tool_depth", default=0)

_DelegateHandler = Callable[..., Any]


@dataclass(frozen=True)
class DelegateToolSpec:
    name: str
    description: str
    instructions: str
    system_prompt: str | None
    tool_names: tuple[str, ...]
    output_type: Any | None = None
    timeout_s: float = DELEGATE_TOOL_TIMEOUT_S
    max_depth: int = DELEGATE_TOOL_DEFAULT_DEPTH
    log_context: str | None = None
    include_delegate_tools: bool = True


DELEGATE_TOOL_HANDLERS: dict[str, _DelegateHandler] = {}
DELEGATE_TOOL_ARG_MODELS: dict[str, type[Any]] = {}
DELEGATE_TOOL_DESCRIPTIONS: dict[str, str] = {}
DELEGATE_TOOL_TIMEOUTS: dict[str, float] = {}


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


def _build_delegate_prompt(task: str, context: str | None) -> str:
    task_text = task.strip()
    if context:
        return f"{context.strip()}\n\nTask: {task_text}"
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


def _build_delegate_agent(spec: DelegateToolSpec) -> Any:
    load_dotenv(ENV_FILE, override=False)
    load_dotenv()
    config = load_models_config()
    model_id = model_registry.current_model_id()
    models_cfg = config.get("models", {})
    if model_id not in models_cfg:
        raise ValueError(f"Unknown model id: {model_id}")
    model_entry = models_cfg.get(model_id, {})

    model_obj, model_settings = _build_provider_model(model_id, model_entry)
    agent = PydanticAgent(
        model_obj,
        toolsets=(),
        system_prompt=spec.system_prompt or SYSTEM_PROMPT,
        instructions=spec.instructions,
        model_settings=model_settings,
        output_type=spec.output_type,
    )

    from isaac.agent.tools import TOOL_HANDLERS, register_tools

    tool_names = _expand_tool_names(spec)
    unknown = [name for name in tool_names if name not in TOOL_HANDLERS]
    if unknown:
        raise ValueError(f"Unknown delegate tool(s): {', '.join(sorted(unknown))}")
    register_tools(agent, tool_names=tool_names)
    return agent


async def run_delegate_tool(
    spec: DelegateToolSpec,
    *,
    task: str,
    context: str | None = None,
) -> dict[str, Any]:
    """Run a delegate agent tool with depth guarding."""
    depth = _delegate_depth.get()
    if depth >= spec.max_depth:
        return {
            "content": "",
            "error": f"Delegate tool depth exceeded (max {spec.max_depth}).",
        }

    token = _delegate_depth.set(depth + 1)
    try:
        agent = _build_delegate_agent(spec)
        prompt = _build_delegate_prompt(task, context)
        structured: Any | None = None

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

        # Delegate agents start with no shared history by default.
        response, _ = await stream_with_runner(
            agent,
            prompt,
            _noop,
            _noop,
            asyncio.Event(),
            on_event=_capture,
            log_context=spec.log_context,
        )
        if response is None:
            return {"content": "", "error": "Delegate agent cancelled."}
        if response.startswith("Provider error:"):
            msg = response.removeprefix("Provider error:").strip()
            return {"content": "", "error": msg}
        if response.lower().startswith("model output failed validation"):
            return {"content": "", "error": response}

        output = structured or response
        return {"content": output, "error": None}
    except Exception as exc:
        return {"content": "", "error": str(exc)}
    finally:
        _delegate_depth.reset(token)
