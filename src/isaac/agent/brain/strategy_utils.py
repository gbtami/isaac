"""Shared utilities for prompt strategies."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from pydantic_ai import Agent as PydanticAgent  # type: ignore

from acp.helpers import update_plan
from acp.schema import PlanEntry

from isaac.agent.brain.prompt import (
    EXECUTOR_INSTRUCTIONS,
    PLANNER_INSTRUCTIONS,
    SUBAGENT_INSTRUCTIONS,
    TODO_PLANNER_INSTRUCTIONS,
    SYSTEM_PROMPT,
)
from isaac.agent.brain.strategy_plan import PlanSteps
from isaac.agent.brain.planner import parse_plan_from_text
from isaac.agent.tools import register_readonly_tools
from isaac.agent.models import ENV_FILE, load_models_config, _build_provider_model


def create_agents_for_model(
    model_id: str, register_tools: Any, toolsets: list[Any] | None = None, system_prompt: str | None = None
) -> tuple[Any, Any]:
    """Build executor and planner agents for a model id."""

    load_dotenv(ENV_FILE, override=False)
    load_dotenv()
    config = load_models_config()
    models_cfg = config.get("models", {})
    if model_id not in models_cfg:
        raise ValueError(f"Unknown model id: {model_id}")
    model_entry = models_cfg.get(model_id, {})

    model_obj, model_settings = _build_provider_model(model_id, model_entry)
    planner_obj, planner_settings = _build_provider_model(model_id, model_entry)

    executor = PydanticAgent(
        model_obj,
        toolsets=toolsets or (),
        system_prompt=system_prompt or SYSTEM_PROMPT,
        instructions=EXECUTOR_INSTRUCTIONS,
        model_settings=model_settings,
    )
    register_tools(executor)

    planner = PydanticAgent(
        planner_obj,
        system_prompt=system_prompt or SYSTEM_PROMPT,
        instructions=PLANNER_INSTRUCTIONS,
        model_settings=planner_settings,
        toolsets=(),
        output_type=PlanSteps,
    )
    register_readonly_tools(planner)
    return executor, planner


def create_subagent_for_model(
    model_id: str, register_tools: Any, toolsets: list[Any] | None = None, system_prompt: str | None = None
) -> Any:
    """Build a single-runner agent for subagent mode."""

    load_dotenv(ENV_FILE, override=False)
    load_dotenv()
    config = load_models_config()
    models_cfg = config.get("models", {})
    if model_id not in models_cfg:
        raise ValueError(f"Unknown model id: {model_id}")
    model_entry = models_cfg.get(model_id, {})

    model_obj, model_settings = _build_provider_model(model_id, model_entry)

    runner = PydanticAgent(
        model_obj,
        toolsets=toolsets or (),
        system_prompt=system_prompt or SYSTEM_PROMPT,
        instructions=SUBAGENT_INSTRUCTIONS,
        model_settings=model_settings,
    )
    register_tools(runner)
    return runner


def create_subagent_planner_for_model(model_id: str, system_prompt: str | None = None) -> Any:
    """Build a delegate planner agent for the subagent todo tool."""

    load_dotenv(ENV_FILE, override=False)
    load_dotenv()
    config = load_models_config()
    models_cfg = config.get("models", {})
    if model_id not in models_cfg:
        raise ValueError(f"Unknown model id: {model_id}")
    model_entry = models_cfg.get(model_id, {})

    planner_obj, planner_settings = _build_provider_model(model_id, model_entry)

    planner = PydanticAgent(
        planner_obj,
        system_prompt=system_prompt or SYSTEM_PROMPT,
        instructions=TODO_PLANNER_INSTRUCTIONS,
        model_settings=planner_settings,
        toolsets=(),
        output_type=PlanSteps,
    )
    register_readonly_tools(planner)
    return planner


def prepare_executor_prompt(
    prompt_text: str, *, plan_update: Any | None = None, plan_response: str | None = None
) -> str:
    plan_lines = [getattr(e, "content", "") for e in getattr(plan_update, "entries", []) or []]
    if plan_lines:
        plan_block = "\n".join(f"- {line}" for line in plan_lines if line)
        return f"{prompt_text}\n\nPlan:\n{plan_block}\n\n{EXECUTOR_INSTRUCTIONS}"
    if plan_response:
        return f"{prompt_text}\n\nPlan:\n{plan_response}\n\n{EXECUTOR_INSTRUCTIONS}"
    return prompt_text


def plan_with_status(plan_update: Any, *, active_index: int | None = None, status_all: str | None = None) -> Any:
    try:
        entries = list(getattr(plan_update, "entries", []) or [])
    except Exception:
        return plan_update
    if not entries:
        return plan_update
    updated: list[PlanEntry] = []
    for idx, entry in enumerate(entries):
        if status_all is not None:
            status = status_all
        elif active_index is not None:
            if idx < active_index:
                status = "completed"
            elif idx == active_index:
                status = "in_progress"
            else:
                status = "pending"
        else:
            status = getattr(entry, "status", "pending")
        try:
            updated.append(entry.model_copy(update={"status": status}))
        except Exception:
            updated.append(entry)
    try:
        return update_plan(updated)
    except Exception:
        return plan_update


def plan_update_from_steps(steps: list[Any]) -> Any | None:
    """Build a plan update with stable IDs from structured PlanStep objects."""

    entries: list[PlanEntry] = []
    for idx, item in enumerate(steps):
        content = getattr(item, "content", "")
        if not isinstance(content, str) or not content.strip():
            continue
        priority = getattr(item, "priority", "medium")
        if priority not in {"high", "medium", "low"}:
            priority = "medium"
        step_id = getattr(item, "id", None) or f"step_{idx + 1}_{uuid4().hex[:6]}"
        try:
            entry = PlanEntry(content=content.strip(), priority=priority, status="pending")
            entry = entry.model_copy(update={"field_meta": {"id": step_id}})
            entries.append(entry)
        except Exception:
            continue
    if not entries:
        return None
    try:
        return update_plan(entries)
    except Exception:
        return None


def coerce_tool_args(raw_args: Any) -> dict[str, Any]:
    """Convert tool call args to a dict, handling common non-dict shapes."""

    if raw_args is None:
        return {}
    if isinstance(raw_args, dict):
        return dict(raw_args)
    if isinstance(raw_args, str):
        return {"command": raw_args}

    for attr in ("model_dump", "dict"):
        func = getattr(raw_args, attr, None)
        if callable(func):
            try:
                data = func()
                if isinstance(data, dict):
                    return dict(data)
            except Exception:
                continue

    collected: dict[str, Any] = {}
    for key in ("command", "cwd", "timeout"):
        if hasattr(raw_args, key):
            try:
                collected[key] = getattr(raw_args, key)
            except Exception:
                continue

    if collected:
        return collected

    try:
        mapping = dict(raw_args)  # type: ignore[arg-type]
        if isinstance(mapping, dict):
            return mapping
    except Exception:
        return {}
    return {}


def parse_or_build_plan(plan_obj: Any) -> Any | None:
    """Parse plan content into update_plan structure."""

    if isinstance(plan_obj, str):
        return parse_plan_from_text(plan_obj or "")
    if isinstance(plan_obj, dict):
        return parse_plan_from_text(str(plan_obj))
    return None
