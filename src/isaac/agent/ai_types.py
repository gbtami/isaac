"""Shared pydantic-ai typing aliases."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias

from pydantic_ai import Agent as PydanticAgent  # type: ignore
from pydantic_ai import RunContext  # type: ignore
from pydantic_ai.models import Model  # type: ignore
from pydantic_ai.settings import ModelSettings  # type: ignore


AgentRunner: TypeAlias = PydanticAgent[Any, Any]
ToolContext: TypeAlias = RunContext[Any]
ModelLike: TypeAlias = Model
ModelSettingsLike: TypeAlias = ModelSettings | None


@dataclass(frozen=True)
class SessionToolDeps:
    """Runtime-only context for model-triggered Isaac tools.

    Pydantic AI exposes this via ``RunContext.deps``. It deliberately is not
    part of the model-facing tool schema: models should call ``read_file`` or
    ``run_command`` with project-relative paths while Isaac binds those calls to
    the active ACP session workspace here.
    """

    session_id: str
    cwd: Path
    additional_directories: tuple[Path, ...] = field(default_factory=tuple)
    mode: str = "ask"
    model_id: str = ""


__all__ = ["AgentRunner", "ToolContext", "ModelLike", "ModelSettingsLike", "SessionToolDeps"]
