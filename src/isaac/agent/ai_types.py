"""Shared pydantic-ai typing aliases."""

from __future__ import annotations

from typing import Any, Callable, TypeAlias

from pydantic_ai import Agent as PydanticAgent  # type: ignore
from pydantic_ai import RunContext  # type: ignore
from pydantic_ai.models import Model  # type: ignore
from pydantic_ai.settings import ModelSettings  # type: ignore


AgentRunner: TypeAlias = PydanticAgent[Any, Any]
ToolContext: TypeAlias = RunContext[Any]
ModelLike: TypeAlias = Model
ModelSettingsLike: TypeAlias = ModelSettings | None
ToolRegister: TypeAlias = Callable[[AgentRunner], None]

__all__ = ["AgentRunner", "ToolContext", "ModelLike", "ModelSettingsLike", "ToolRegister"]
