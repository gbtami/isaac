"""Isaac default ACP configuration helpers."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from acp.schema import (
    AgentCapabilities,
    McpCapabilities,
    PromptCapabilities,
    SessionCapabilities,
    SessionListCapabilities,
)

from isaac.acp.core import (
    ACPAgentConfig,
    FileSystemBackend,
    ModelRegistry,
    PlanSupport,
    PromptStrategy,
    SlashCommands,
    TerminalBackend,
    ToolIO,
    UsageFormatter,
)
from isaac.agent import models as model_registry
from isaac.agent.agent_terminal import (
    create_terminal,
    kill_terminal,
    release_terminal,
    terminal_output,
    wait_for_terminal_exit,
)
from isaac.agent.brain.prompt import SYSTEM_PROMPT
from isaac.agent.brain.strategy_base import ModelBuildError
from isaac.agent.brain.strategy_runner import StrategyEnv
from isaac.agent.brain.subagent_strategy import SubagentPromptStrategy
from isaac.agent.constants import TOOL_OUTPUT_LIMIT
from isaac.agent.fs import read_text_file, write_text_file
from isaac.agent.mcp_support import build_mcp_toolsets
from isaac.agent.planner import build_plan_notification, parse_plan_request
from isaac.agent.prompt_utils import coerce_user_text, extract_prompt_text
from isaac.agent.session_modes import build_mode_state
from isaac.agent.session_store import SessionStore
from isaac.agent.slash import available_slash_commands, handle_slash_command
from isaac.agent.tool_io import await_with_cancel, truncate_text, truncate_tool_output
from isaac.agent.tools import register_tools, run_tool
from isaac.agent.usage import format_usage_summary, normalize_usage

DEFAULT_COMMAND_TIMEOUT_S = 30.0


def build_isaac_capabilities() -> AgentCapabilities:
    """Return the default ACP capabilities for Isaac."""
    return AgentCapabilities(
        load_session=True,
        prompt_capabilities=PromptCapabilities(
            embedded_context=True,
            image=False,
            audio=False,
        ),
        mcp_capabilities=McpCapabilities(http=True, sse=True),
        session_capabilities=SessionCapabilities(list=SessionListCapabilities()),
    )


def build_isaac_system_prompt(cwd: Path) -> str | None:
    """Return Isaac's system prompt with AGENTS.md prepended if present."""
    base_prompt = SYSTEM_PROMPT
    try:
        agents_path = cwd / "AGENTS.md"
        if agents_path.is_file():
            return f"{base_prompt}\n\n{agents_path.read_text(encoding='utf-8', errors='ignore')}"
    except Exception:
        return base_prompt
    return base_prompt


def build_isaac_prompt_strategy(agent: object) -> PromptStrategy:
    """Return Isaac's default prompt strategy."""
    env = StrategyEnv(
        session_modes=getattr(agent, "_session_modes", {}),
        session_last_chunk=getattr(agent, "_session_last_chunk", {}),
        send_update=getattr(agent, "_send_update"),
        request_run_permission=lambda session_id, tool_call_id, command, cwd: getattr(agent, "_request_run_permission")(
            session_id=session_id,
            tool_call_id=tool_call_id,
            command=command,
            cwd=cwd,
        ),
        set_usage=lambda session_id, usage: getattr(agent, "_set_usage")(session_id, usage),
    )
    return SubagentPromptStrategy(
        env=env,
        register_tools=register_tools,
    )


def build_isaac_config(
    *,
    agent_name: str = "isaac",
    agent_title: str = "Isaac ACP Agent",
    agent_version: str = "0.1.0",
    prompt_strategy: PromptStrategy | None = None,
) -> ACPAgentConfig:
    """Build the Isaac default ACPAgentConfig for reuse in other agents."""
    return ACPAgentConfig(
        agent_name=agent_name,
        agent_title=agent_title,
        agent_version=agent_version,
        capabilities_builder=build_isaac_capabilities,
        build_system_prompt=build_isaac_system_prompt,
        prompt_strategy=prompt_strategy,
        prompt_strategy_factory=None if prompt_strategy is not None else build_isaac_prompt_strategy,
        build_mcp_toolsets=build_mcp_toolsets,
        slash_commands=SlashCommands(
            available_commands=available_slash_commands,
            handle_command=handle_slash_command,
        ),
        plan_support=PlanSupport(
            parse_request=parse_plan_request,
            build_notification=build_plan_notification,
        ),
        model_registry=ModelRegistry(
            current_model_id=model_registry.current_model_id,
            set_current_model=model_registry.set_current_model,
            list_models=model_registry.list_user_models,
            get_context_limit=model_registry.get_context_limit,
        ),
        usage_formatter=UsageFormatter(
            normalize_usage=normalize_usage,
            format_usage_summary=format_usage_summary,
        ),
        session_store=SessionStore(Path.home() / ".isaac" / "sessions"),
        file_system=FileSystemBackend(
            read_text_file=read_text_file,
            write_text_file=write_text_file,
        ),
        terminal=TerminalBackend(
            create_terminal=create_terminal,
            terminal_output=terminal_output,
            wait_for_terminal_exit=wait_for_terminal_exit,
            kill_terminal=kill_terminal,
            release_terminal=release_terminal,
        ),
        tool_runner=lambda *args, **kwargs: run_tool(*args, **kwargs),
        tool_io=ToolIO(
            truncate_text=truncate_text,
            truncate_tool_output=truncate_tool_output,
            await_with_cancel=await_with_cancel,
        ),
        build_mode_state=build_mode_state,
        model_build_error=ModelBuildError,
        tool_output_limit=TOOL_OUTPUT_LIMIT,
        terminal_output_limit=TOOL_OUTPUT_LIMIT,
        command_timeout_s=DEFAULT_COMMAND_TIMEOUT_S,
        extract_prompt_text=extract_prompt_text,
        coerce_user_text=coerce_user_text,
    )


def apply_overrides(config: ACPAgentConfig, **overrides: Any) -> ACPAgentConfig:
    """Return a copy of the config with selected fields overridden."""
    return replace(config, **overrides)
