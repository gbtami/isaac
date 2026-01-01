"""Internal prompt result shape shared by prompt handlers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


PromptStopReason = Literal["end_turn", "cancelled", "refusal"]


@dataclass(frozen=True)
class PromptResult:
    """Protocol-agnostic prompt response used by the brain layer."""

    stop_reason: PromptStopReason


__all__ = ["PromptResult", "PromptStopReason"]
