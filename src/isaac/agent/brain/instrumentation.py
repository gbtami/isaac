"""Instrumentation helpers for pydantic-ai integration."""

from __future__ import annotations

import os
from typing import Any


def pydantic_ai_instrument_enabled() -> bool:
    """Return whether pydantic-ai instrumentation should be enabled."""

    raw = os.getenv("ISAAC_PYDANTIC_AI_INSTRUMENT", "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    return False


def base_run_metadata(*, component: str, model_id: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build stable metadata attached to pydantic-ai runs."""

    metadata: dict[str, Any] = {"component": component, "model_id": model_id}
    if extra:
        metadata.update(extra)
    return metadata
