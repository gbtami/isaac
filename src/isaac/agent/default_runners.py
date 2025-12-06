"""Fallback model runners used when user-configured models fail to load."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from isaac.agent import models as model_registry
from isaac.agent.runner import register_tools

logger = logging.getLogger("acp_server")


@dataclass
class SimpleRunResult:
    output: str


class SimpleAIRunner:
    async def run(self, prompt: str) -> SimpleRunResult:
        return SimpleRunResult(output=f"Echo: {prompt}")

    async def run_stream_events(self, prompt: str):
        yield f"Echo: {prompt}"


def create_default_runners(toolsets: list[Any] | None = None) -> tuple[Any, Any]:
    """Build executor/planner runners, falling back to simple echo runner on errors."""

    try:
        config = model_registry.load_models_config()
        current = config.get("current", "test")
        return model_registry.build_agent_pair(current, register_tools, toolsets=toolsets or [])
    except Exception as exc:  # pragma: no cover - fallback when model creation fails
        logger.warning("Falling back to simple runner: %s", exc)
        runner = SimpleAIRunner()
        return runner, runner
