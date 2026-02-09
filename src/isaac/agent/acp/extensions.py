"""ACP extension method handlers."""

from __future__ import annotations

import logging
from typing import Any

from isaac.log_utils import log_context, log_event

logger = logging.getLogger(__name__)


class ExtensionsMixin:
    async def ext_method(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Handle extension methods."""
        with log_context(session_id=payload.get("session_id"), ext_method=name):
            log_event(logger, "acp.ext.request", params_keys=sorted(payload.keys()))
        return {"error": f"Unknown ext method: {name}"}

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        """Handle extension notifications (noop placeholder to satisfy ACP interface)."""
        with log_context(ext_method=method):
            log_event(logger, "acp.ext.notification", params_keys=sorted(params.keys()))
