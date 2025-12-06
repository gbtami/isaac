"""Utilities for working with prompt blocks and plan-only hints."""

from __future__ import annotations

from typing import Any


def extract_prompt_text(blocks: list[Any]) -> str:
    parts: list[str] = []
    for block in blocks:
        text_val = getattr(block, "text", None)
        if text_val:
            parts.append(text_val)
            continue
        resource = getattr(block, "resource", None)
        if resource and hasattr(resource, "text") and getattr(resource, "text", None):
            parts.append(str(resource.text))
            continue
        uri = getattr(block, "uri", None)
        if uri:
            parts.append(f"[resource:{uri}]")
    return "".join(parts)


def coerce_user_text(block: Any) -> str | None:
    """Extract a text string from arbitrary user prompt blocks."""

    if block is None:
        return None
    if isinstance(block, str):
        return block
    # dict payload with text
    if isinstance(block, dict):
        if "text" in block and isinstance(block["text"], str):
            return block["text"]
    text_val = getattr(block, "text", None)
    if isinstance(text_val, str):
        return text_val
    resource = getattr(block, "resource", None)
    if resource and hasattr(resource, "text"):
        text = getattr(resource, "text", None)
        if isinstance(text, str):
            return text
    return None


def is_plan_only_prompt(prompt_text: str) -> bool:
    """Heuristic to avoid executing when the user explicitly asked for a plan only."""

    lowered = prompt_text.lower()
    if lowered.startswith("plan:"):
        return True
    plan_only_markers = [
        "plan only",
        "only a plan",
        "just a plan",
        "just planning",
        "planning only",
    ]
    return any(marker in lowered for marker in plan_only_markers)
