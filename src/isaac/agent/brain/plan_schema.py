"""Plan schemas used by prompt handling."""

from __future__ import annotations

from typing import Any, List, Literal

from pydantic import BaseModel, model_validator


class PlanStep(BaseModel):
    content: str
    priority: Literal["high", "medium", "low"] = "medium"
    id: str | None = None


class PlanSteps(BaseModel):
    entries: List[PlanStep]

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_formats(cls, value: Any) -> Any:
        """Accept older/looser plan shapes and normalize to entries."""
        entries: Any = None
        if isinstance(value, dict):
            if "entries" in value:
                entries = value.get("entries")
            elif "steps" in value:
                entries = value.get("steps")
        else:
            entries = value

        if isinstance(entries, list):
            normalized: list[dict[str, Any]] = []
            for item in entries:
                if isinstance(item, PlanStep):
                    normalized.append(item.model_dump())
                elif isinstance(item, str):
                    text = item.strip()
                    if text:
                        normalized.append({"content": text})
                elif isinstance(item, dict):
                    if "content" not in item and "step" in item:
                        item = {**item, "content": item.get("step")}
                    normalized.append(item)
            return {"entries": normalized}

        if isinstance(entries, str):
            lines = [ln.strip() for ln in entries.splitlines() if ln.strip()]
            if not lines:
                return {"entries": []}
            items: list[str] = []
            for ln in lines:
                cleaned = ln.lstrip("-* ").strip()
                if cleaned and cleaned[0].isdigit() and "." in cleaned:
                    cleaned = cleaned.split(".", 1)[1].strip()
                if cleaned:
                    items.append(cleaned)
            return {"entries": [{"content": it} for it in items]} if items else {"entries": []}

        return value


__all__ = ["PlanStep", "PlanSteps"]
