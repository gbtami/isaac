"""Planning helpers for a single-agent approach.

The model is prompted to emit a short plan (e.g., markdown list). This module
parses that output into ACP plan entries when present.
"""

from __future__ import annotations

from typing import List

from acp.helpers import plan_entry, update_plan
from acp.schema import AgentPlanUpdate


def parse_plan_from_text(output: str) -> AgentPlanUpdate | None:
    """Parse a simple markdown/numbered list into an ACP AgentPlanUpdate.

    A plan is detected if the text contains a line starting with "Plan:" or
    a numbered/bullet list. This is intentionally simple and can be tightened
    if needed.
    """
    lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
    if not lines:
        return None

    # Detect an explicit Plan header
    start = 0
    for idx, line in enumerate(lines):
        if line.lower().startswith("plan:"):
            start = idx + 1
            break

    candidates: List[str] = []
    for line in lines[start:]:
        if line[0] in {"-", "*"}:
            candidates.append(line.lstrip("-* "))
        elif line[:2].isdigit() and "." in line:
            candidates.append(line.split(".", 1)[1].strip())

    if not candidates:
        return None

    entries = [plan_entry(item) for item in candidates if item]
    if not entries:
        return None
    return update_plan(entries)
