"""Coercion helpers for tool call arguments."""

from __future__ import annotations

import json
from typing import Any


def coerce_tool_args(raw_args: Any) -> dict[str, Any]:
    """Convert tool call args to a dict, handling common non-dict shapes.

    This normalizes JSON-encoded strings so tool-call metadata like `path`
    and `task` survives into history and logging.
    """

    if raw_args is None:
        return {}
    if isinstance(raw_args, dict):
        return dict(raw_args)
    if isinstance(raw_args, str):
        candidate = raw_args.strip()
        if candidate:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                return parsed
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
