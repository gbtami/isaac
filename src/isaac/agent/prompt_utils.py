"""Utilities for working with prompt blocks."""

from __future__ import annotations

from typing import Any

from acp.schema import EmbeddedResourceContentBlock, ResourceContentBlock


def extract_prompt_text(blocks: list[Any]) -> str:
    parts: list[str] = []
    for block in blocks:
        text_val = getattr(block, "text", None)
        if text_val:
            parts.append(text_val)
            continue
        resource = getattr(block, "resource", None)
        if resource:
            text = getattr(resource, "text", None)
            if isinstance(text, str) and text:
                parts.append(text)
                continue
            uri = getattr(resource, "uri", None)
            if uri:
                parts.append(f"[resource:{uri}]")
                continue
        if isinstance(block, ResourceContentBlock):
            if block.uri:
                parts.append(f"[resource:{block.uri}]")
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
    if isinstance(block, EmbeddedResourceContentBlock):
        res = getattr(block, "resource", None)
        if res and isinstance(getattr(res, "text", None), str):
            return getattr(res, "text")
    return None
