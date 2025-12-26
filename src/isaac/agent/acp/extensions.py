"""ACP extension method handlers."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("acp_server")


class ExtensionsMixin:
    async def ext_method(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Handle extension methods for model listing/selection."""
        if name == "model/list":
            session_id = payload.get("session_id")
            current = self._session_model_ids.get(session_id, self._current_model_id())
            models = self._list_user_models()
            return {
                "current": current,
                "models": [{"id": mid, "description": meta.get("description", "")} for mid, meta in models.items()],
            }
        if name == "model/set":
            session_id = payload.get("session_id")
            model_id = payload.get("model_id")
            if not session_id or not model_id:
                return {"error": "session_id and model_id required"}
            try:
                await self.set_session_model(model_id, session_id)
                self._session_model_ids[session_id] = model_id
                return {"current": model_id}
            except Exception as exc:  # noqa: BLE001
                return {"error": str(exc)}
        return {"error": f"Unknown ext method: {name}"}

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        """Handle extension notifications (noop placeholder to satisfy ACP interface)."""
        logger.info("Received ext notification %s params_keys=%s", method, sorted(params.keys()))

    @staticmethod
    def _list_user_models() -> dict[str, Any]:
        from isaac.agent import models as model_registry

        return model_registry.list_user_models()
