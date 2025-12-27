"""Delegate sub-agent tooling and registries."""

from __future__ import annotations

from importlib import import_module
import pkgutil

from isaac.agent.subagents.delegate_tools import (
    DELEGATE_TOOL_ARG_MODELS,
    DELEGATE_TOOL_DESCRIPTIONS,
    DELEGATE_TOOL_HANDLERS,
    DELEGATE_TOOL_TIMEOUTS,
    DelegateToolSpec,
    register_delegate_tool,
    run_delegate_tool,
)

_DISCOVERY_SKIP = {"__init__", "delegate_tools", "args", "outputs"}
_DISCOVERED = False


def _discover_delegate_tools() -> None:
    """Import sub-agent modules so they can register their delegate tools."""

    global _DISCOVERED
    if _DISCOVERED:
        return
    _DISCOVERED = True

    for module in pkgutil.iter_modules(__path__):  # type: ignore[name-defined]
        name = module.name
        if name in _DISCOVERY_SKIP or name.startswith("_"):
            continue
        import_module(f"{__name__}.{name}")


_discover_delegate_tools()

__all__ = [
    "DELEGATE_TOOL_ARG_MODELS",
    "DELEGATE_TOOL_DESCRIPTIONS",
    "DELEGATE_TOOL_HANDLERS",
    "DELEGATE_TOOL_TIMEOUTS",
    "DelegateToolSpec",
    "register_delegate_tool",
    "run_delegate_tool",
]
