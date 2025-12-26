"""Session lifecycle handlers for ACP sessions."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from pathlib import Path
from typing import Any

from acp import LoadSessionResponse, NewSessionResponse, SetSessionModeResponse, SetSessionModelResponse
from acp.schema import ListSessionsResponse, SessionInfo, SessionNotification

from isaac.agent import models as model_registry
from isaac.agent.brain.model_errors import ModelBuildError
from isaac.agent.brain.prompt import SYSTEM_PROMPT
from isaac.agent.mcp_support import build_mcp_toolsets
from isaac.agent.session_modes import build_mode_state

logger = logging.getLogger("acp_server")


class SessionLifecycleMixin:
    def _build_session_system_prompt(self, cwd: Path) -> str | None:
        """Build a per-session system prompt with AGENTS.md prepended if present."""

        base_prompt = SYSTEM_PROMPT
        try:
            agents_path = cwd / "AGENTS.md"
            if agents_path.is_file():
                return f"{base_prompt}\n\n{agents_path.read_text(encoding='utf-8', errors='ignore')}"
        except Exception:
            return base_prompt
        return base_prompt

    async def new_session(
        self,
        cwd: str,
        mcp_servers: list[Any],
        **_: Any,
    ) -> NewSessionResponse:
        """Create a new session (Session Setup / creation)."""
        session_id = str(uuid.uuid4())
        logger.info("Received new session request: %s cwd=%s", session_id, cwd)
        self._sessions.add(session_id)
        cwd_path = self._require_absolute_cwd(cwd)
        self._session_cwds[session_id] = cwd_path
        session_system_prompt = self._build_session_system_prompt(cwd_path)
        self._session_system_prompts[session_id] = session_system_prompt
        self._cancel_events[session_id] = asyncio.Event()
        toolsets = build_mcp_toolsets(mcp_servers)
        self._session_toolsets[session_id] = toolsets
        await self._prompt_handler.init_session(session_id, toolsets, system_prompt=session_system_prompt)
        self._session_history[session_id] = []
        self._session_allowed_commands[session_id] = set()
        self._session_mcp_servers[session_id] = mcp_servers
        model_id = self._prompt_handler.model_id(session_id) or self._current_model_id()
        self._session_model_ids[session_id] = model_id
        self._persist_session_meta(
            session_id,
            cwd_path,
            mcp_servers,
            current_mode="ask",
        )
        mode_state = build_mode_state(self._session_modes, session_id, current_mode="ask")
        await self._send_available_commands(session_id)
        return NewSessionResponse(session_id=session_id, modes=mode_state)

    async def load_session(
        self,
        cwd: str,
        mcp_servers: list[Any],
        session_id: str,
        **_: Any,
    ) -> LoadSessionResponse | None:
        """Reload an existing session and replay history (Session Setup / loading)."""
        logger.info("Received load session request %s", session_id)
        self._sessions.add(session_id)
        stored_meta = self._load_session_meta(session_id)
        cwd_path = self._require_absolute_cwd(cwd)
        self._session_cwds[session_id] = cwd_path
        session_system_prompt = self._build_session_system_prompt(cwd_path)
        self._session_system_prompts[session_id] = session_system_prompt
        self._cancel_events.setdefault(session_id, asyncio.Event())
        mcp_servers = mcp_servers or stored_meta.get("mcpServers", [])
        toolsets = build_mcp_toolsets(mcp_servers)
        self._session_toolsets[session_id] = toolsets
        await self._prompt_handler.init_session(session_id, toolsets, system_prompt=session_system_prompt)
        self._session_history.setdefault(session_id, [])
        self._session_allowed_commands.setdefault(session_id, set())
        self._session_mcp_servers[session_id] = mcp_servers
        mode = stored_meta.get("mode") or self._session_modes.get(session_id, "ask")
        self._session_modes[session_id] = mode
        model_id = self._prompt_handler.model_id(session_id) or self._current_model_id()
        self._session_model_ids[session_id] = model_id
        history = self._load_session_history(session_id)
        self._session_history[session_id].extend(history)
        self._persist_session_meta(
            session_id,
            cwd_path,
            mcp_servers,
            current_mode=mode,
        )
        for note in history:
            await self._conn.session_update(session_id=note.session_id, update=note.update)  # type: ignore[arg-type]
        await self._send_available_commands(session_id)
        return LoadSessionResponse()

    async def list_sessions(self, cursor: str | None = None, cwd: str | None = None, **_: Any) -> ListSessionsResponse:
        """Return known sessions; minimal implementation without paging."""
        sessions: list[SessionInfo] = []
        for session_id in self._sessions:
            sessions.append(
                SessionInfo(
                    session_id=session_id,
                    cwd=str(self._session_cwds.get(session_id, cwd or Path.cwd())),
                )
            )
        return ListSessionsResponse(sessions=sessions, next_cursor=None)

    async def set_session_mode(self, mode_id: str, session_id: str, **_: Any) -> SetSessionModeResponse | None:
        """Update the current session mode and broadcast (Session Modes)."""
        logger.info(
            "Received set session mode request %s -> %s",
            session_id,
            mode_id,
        )
        self._session_modes[session_id] = mode_id
        await self._send_update(self._mode_update(session_id, mode_id))
        self._persist_session_meta(
            session_id,
            self._session_cwds.get(session_id, Path.cwd()),
            self._session_mcp_servers.get(session_id, []),
            current_mode=mode_id,
        )
        return SetSessionModeResponse()

    async def set_session_model(self, model_id: str, session_id: str, **_: Any) -> SetSessionModelResponse | None:
        """Switch the backing model for a session."""
        logger.info("Received set session model request %s -> %s", session_id, model_id)
        previous_model_id = self._session_model_ids.get(session_id, model_registry.current_model_id())
        try:
            await self._prompt_handler.set_session_model(
                session_id,
                model_id,
                toolsets=self._session_toolsets.get(session_id, []),
                system_prompt=self._session_system_prompts.get(session_id),
            )
            self._session_model_ids[session_id] = model_id
            with contextlib.suppress(Exception):
                model_registry.set_current_model(model_id)
        except ModelBuildError:
            with contextlib.suppress(Exception):
                model_registry.set_current_model(previous_model_id)
            raise
        return SetSessionModelResponse()

    def _mode_update(self, session_id: str, mode_id: str) -> SessionNotification:
        from acp.helpers import session_notification
        from acp.schema import CurrentModeUpdate

        return session_notification(
            session_id,
            CurrentModeUpdate(session_update="current_mode_update", current_mode_id=mode_id),
        )

    def _current_model_id(self) -> str:
        return model_registry.current_model_id()

    def _persist_session_meta(
        self,
        session_id: str,
        cwd: Path,
        mcp_servers: list[Any] | None,
        *,
        current_mode: str,
    ) -> None:
        self._session_store.persist_meta(session_id, cwd, mcp_servers or [], current_mode=current_mode)

    def _load_session_meta(self, session_id: str) -> dict[str, Any]:
        return self._session_store.load_meta(session_id)

    def _load_session_history(self, session_id: str) -> list[SessionNotification]:
        return self._session_store.load_history(session_id)

    @staticmethod
    def _require_absolute_cwd(cwd: str) -> Path:
        from acp import RequestError

        path = Path(cwd or "").expanduser()
        if not path.is_absolute():
            raise RequestError.invalid_request({"message": "cwd must be an absolute path"})
        return path

    async def _send_available_commands(self, session_id: str) -> None:
        """Advertise slash commands using `available_commands_update` per ACP spec."""
        if session_id in self._session_commands_advertised:
            return
        commands = self._available_slash_commands()
        update = self._available_commands_update(session_id, commands)
        await self._send_update(update)
        self._session_commands_advertised.add(session_id)

    @staticmethod
    def _available_slash_commands() -> list[Any]:
        from isaac.agent.slash import available_slash_commands

        return available_slash_commands()

    def _available_commands_update(self, session_id: str, commands: list[Any]) -> SessionNotification:
        from acp.helpers import session_notification
        from acp.schema import AvailableCommandsUpdate

        update = AvailableCommandsUpdate(session_update="available_commands_update", available_commands=commands)
        return session_notification(session_id, update)
