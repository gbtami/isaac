"""Parse and emit ACP plan updates from inline prompt shortcuts.

See: https://agentclientprotocol.com/protocol/agent-plan
"""

from __future__ import annotations

from typing import List

from acp import SessionNotification
from acp.helpers import plan_entry, update_plan
from acp.schema import AgentPlanContentUpdate, PlanUpdateItems

from isaac.agent.acp.plan_updates import DEFAULT_PLAN_ID


def parse_plan_shortcut(prompt_text: str) -> list[str] | None:
    if not prompt_text.startswith("plan:"):
        return None
    content = prompt_text[len("plan:") :].strip()
    if not content:
        return None
    if ";" in content:
        return [part.strip() for part in content.split(";")]
    if "," in content:
        return [part.strip() for part in content.split(",")]
    return [content]


def build_plan_shortcut_notification(
    session_id: str,
    items: List[str],
    *,
    use_incremental: bool = False,
    plan_id: str = DEFAULT_PLAN_ID,
) -> SessionNotification:
    entries = [plan_entry(item.strip()) for item in items if item.strip()]
    if use_incremental:
        update = AgentPlanContentUpdate(
            session_update="plan_update",
            plan=PlanUpdateItems(type="items", id=plan_id, entries=entries),
        )
    else:
        update = update_plan(entries)
    return SessionNotification(session_id=session_id, update=update)
