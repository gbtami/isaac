"""Planning helpers for a single-agent approach.

The model is prompted to emit a short plan (e.g., markdown list). This module
parses that output into ACP plan entries when present.
"""

from __future__ import annotations

import re
from typing import List

from acp.helpers import plan_entry, update_plan
from acp.schema import AgentPlanUpdate


def parse_plan_from_text(output: str) -> AgentPlanUpdate | None:
    """Parse a simple markdown/numbered list into an ACP AgentPlanUpdate.

    A plan is preferred when a "Plan:" header is present, but we also accept
    numbered/bulleted lists (with at least two steps) to capture model outputs
    that omit the header.
    """
    lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
    if not lines:
        return None

    start = None
    header_present = False
    header_remainder: str | None = None
    for idx, line in enumerate(lines):
        if line.lower().startswith("plan:"):
            start = idx + 1
            header_present = True
            remainder = line.split(":", 1)[1].strip()
            if remainder:
                header_remainder = remainder
            break

    candidates: List[str] = []
    if header_remainder:
        candidates.extend(_split_inline_items(header_remainder))

    scan_lines = lines[start:] if header_present else lines
    for line in scan_lines:
        if line and line[0] in {"-", "*"}:
            candidates.append(line.lstrip("-* ").strip())
        elif line and line[0].isdigit():
            remainder = line.split(".", 1)
            if len(remainder) == 2:
                candidates.append(remainder[1].strip())
            else:
                candidates.append(line.strip())

    # Without an explicit header, require at least two steps to avoid false positives.
    if not header_present and len(candidates) < 2:
        return None

    if not candidates:
        return None

    entries = [plan_entry(item) for item in candidates if item]
    if not entries:
        return None
    return update_plan(entries)


def _split_inline_items(text: str) -> list[str]:
    """Split inline plan remainder like '1. step one 2. step two' or 'a; b'."""
    parts: list[str] = []
    # First split on semicolons; if none, split on numbered boundaries.
    tokens = re.split(r"[;]", text)
    if len(tokens) == 1:
        tokens = re.split(r"(?=\s*\d+\.)", text)
    for tok in tokens:
        cleaned = tok.strip()
        if not cleaned:
            continue
        if cleaned[0].isdigit() and "." in cleaned:
            cleaned = cleaned.split(".", 1)[1].strip()
        parts.append(cleaned)
    return parts
