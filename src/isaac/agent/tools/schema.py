"""Schema helpers for ACP tool exposure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from isaac.agent.tools.registry import TOOL_ARG_MODELS, TOOL_DESCRIPTIONS, TOOL_HANDLERS


@dataclass
class ToolParameter:
    type: str
    properties: dict[str, Any]
    required: list[str]


@dataclass
class Tool:
    function: str
    description: str
    parameters: ToolParameter


def get_tools() -> List[Any]:
    """Return ACP tool descriptions from pydantic models."""
    base_tools: list[Any] = []
    for name in TOOL_HANDLERS.keys():
        model = TOOL_ARG_MODELS.get(name)
        if model is None:
            continue
        schema = model.model_json_schema()  # type: ignore[attr-defined]
        base_tools.append(
            Tool(
                function=name,
                description=TOOL_DESCRIPTIONS.get(name, ""),
                parameters=ToolParameter(
                    type="object",
                    properties=schema.get("properties", {}),
                    required=schema.get("required", []),
                ),
            )
        )
    return base_tools
