"""Default runner construction without silent fallbacks."""

from __future__ import annotations

from typing import Any, Tuple

from isaac.agent import models as model_registry
from isaac.agent.runner import register_tools


def create_default_runners(toolsets: list[Any] | None = None) -> Tuple[Any, Any]:
    """Build executor/planner runners or let exceptions surface."""

    current = model_registry.current_model_id()
    # Let build_agent_pair raise if misconfigured; callers surface the failure.
    return model_registry.build_agent_pair(current, register_tools, toolsets=toolsets or [])
