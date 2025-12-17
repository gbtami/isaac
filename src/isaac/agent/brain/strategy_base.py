"""Shared prompt strategy interfaces and errors."""

from __future__ import annotations

import asyncio
from typing import Any, Protocol

from acp.schema import PromptResponse


class ModelBuildError(RuntimeError):
    """Raised when a model fails to build for a session."""


class PromptStrategy(Protocol):
    """Interface for per-session prompt processing strategies."""

    async def init_session(self, session_id: str, toolsets: list[Any]) -> None: ...

    async def set_session_model(self, session_id: str, model_id: str, toolsets: list[Any]) -> None: ...

    async def handle_prompt(
        self,
        session_id: str,
        prompt_text: str,
        cancel_event: asyncio.Event,
    ) -> PromptResponse: ...

    def model_id(self, session_id: str) -> str | None: ...
