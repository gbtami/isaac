"""Core ACP-compliant agent implementation.

Contains session/prompt handling, tool execution, filesystem/terminal support,
and ACP lifecycle handlers. Non-protocol conveniences live in agent.py.
"""

from __future__ import annotations

from typing import Any

from isaac.acp.core import ACPAgentCore, PromptStrategy
from isaac.acp.defaults import build_isaac_config
from isaac.agent.tools import run_tool  # noqa: F401


def _tool_runner(*args: Any, **kwargs: Any):
    return run_tool(*args, **kwargs)


class ACPAgent(ACPAgentCore):
    """Implements ACP session, prompt, tool, filesystem, and terminal flows."""

    def __init__(
        self,
        conn: Any | None = None,
        *,
        agent_name: str = "isaac",
        agent_title: str = "Isaac ACP Agent",
        agent_version: str = "0.1.0",
        prompt_strategy: PromptStrategy | None = None,
    ) -> None:
        config = build_isaac_config(
            agent_name=agent_name,
            agent_title=agent_title,
            agent_version=agent_version,
            prompt_strategy=prompt_strategy,
        )
        config.tool_runner = _tool_runner
        super().__init__(config=config, conn=conn)
