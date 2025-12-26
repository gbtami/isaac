"""Session update persistence and replay helpers."""

from __future__ import annotations

from typing import Any

from acp.helpers import session_notification, text_block, update_agent_message
from acp.schema import SessionNotification, UserMessageChunk

from isaac.agent.prompt_utils import coerce_user_text
from isaac.agent.usage import format_usage_summary, normalize_usage


class SessionUpdateMixin:
    async def _send_update(self, note: SessionNotification) -> None:
        """Record and emit a session/update notification for replay support."""
        self._record_update(note)
        if self._conn is None:
            raise RuntimeError("Connection not established")
        sender = getattr(self._conn, "session_update", None)
        if sender is None:
            raise RuntimeError("Connection missing session_update handler")
        await sender(session_id=note.session_id, update=note.update)

    def _record_update(self, note: SessionNotification) -> None:
        """Cache and persist updates for replay after restarts."""
        history = self._session_history.setdefault(note.session_id, [])
        history.append(note)
        self._session_store.persist_update(note.session_id, note)

    def _store_user_prompt(self, session_id: str, prompt_blocks: list[Any]) -> None:
        """Persist user prompt content for session/load replay."""
        for block in prompt_blocks:
            text = coerce_user_text(block)
            if text is None:
                continue
            try:
                chunk = UserMessageChunk(
                    session_update="user_message_chunk",
                    content=text_block(text),
                )
            except Exception:
                continue
            self._record_update(SessionNotification(session_id=session_id, update=chunk))

    async def checkpoint_session(self, session_id: str) -> SessionNotification:
        """Persist prompt state for later restore."""

        snapshot: dict[str, Any] = {}
        handler_snapshot = getattr(self._prompt_handler, "snapshot", None)
        if callable(handler_snapshot):
            snapshot = handler_snapshot(session_id)
        self._session_store.persist_prompt_state(session_id, snapshot)
        return session_notification(
            session_id,
            update_agent_message(text_block("Checkpoint saved.")),
        )

    async def restore_session_state(self, session_id: str) -> SessionNotification:
        """Restore prompt state from persisted snapshot."""

        snapshot = self._session_store.load_prompt_state(session_id)
        if not snapshot:
            return session_notification(
                session_id,
                update_agent_message(text_block("No checkpoint available.")),
            )
        restorer = getattr(self._prompt_handler, "restore_snapshot", None)
        if callable(restorer):
            await restorer(session_id, snapshot)
        return session_notification(
            session_id,
            update_agent_message(text_block("Checkpoint restored.")),
        )

    def _build_usage_note(self, session_id: str) -> SessionNotification:
        """Build a usage summary for the current session on demand."""
        model_id = self._session_model_ids.get(session_id, "")
        context_limit = self._get_context_limit(model_id)
        usage = normalize_usage(self._session_usage.get(session_id))
        summary = format_usage_summary(usage, context_limit, model_id)
        return session_notification(
            session_id,
            update_agent_message(text_block(summary)),
        )

    @staticmethod
    def _get_context_limit(model_id: str) -> int | None:
        from isaac.agent import models as model_registry

        return model_registry.get_context_limit(model_id)
