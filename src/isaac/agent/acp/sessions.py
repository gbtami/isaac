"""Session lifecycle handlers for ACP sessions."""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from pathlib import Path
from typing import Any

from acp import LoadSessionResponse, NewSessionResponse, RequestError, SetSessionModeResponse, SetSessionModelResponse
from acp.schema import (
    ConfigOptionUpdate,
    ListSessionsResponse,
    SessionConfigOption,
    SessionConfigOptionSelect,
    SessionConfigSelectOption,
    SessionInfo,
    SessionNotification,
    SetSessionConfigOptionResponse,
)

from isaac.agent import models as model_registry
from isaac.agent.acp.auth_methods import auth_method_env_var_name, auth_method_payload
from isaac.agent.brain.model_errors import ModelBuildError
from isaac.agent.brain.prompt import SYSTEM_PROMPT
from isaac.agent.mcp_support import build_mcp_toolsets
from isaac.agent.session_modes import available_modes, build_mode_state
from isaac.log_utils import log_context, log_event

logger = logging.getLogger(__name__)


class SessionLifecycleMixin:
    MODE_CONFIG_ID = "mode"
    MODEL_CONFIG_ID = "model"

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
        with log_context(session_id=session_id, cwd=cwd):
            log_event(logger, "acp.session.new")
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
                current_model=model_id,
            )
            mode_state = build_mode_state(self._session_modes, session_id, current_mode="ask")
            await self._send_available_commands(session_id)
            return NewSessionResponse(
                session_id=session_id,
                modes=mode_state,
                config_options=self._session_config_options(session_id),
            )

    async def load_session(
        self,
        cwd: str,
        mcp_servers: list[Any],
        session_id: str,
        **_: Any,
    ) -> LoadSessionResponse | None:
        """Reload an existing session and replay history (Session Setup / loading)."""
        with log_context(session_id=session_id, cwd=cwd):
            log_event(logger, "acp.session.load")
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
        model_id = stored_meta.get("model") or self._prompt_handler.model_id(session_id) or self._current_model_id()
        if model_id != (self._prompt_handler.model_id(session_id) or self._current_model_id()):
            try:
                await self._prompt_handler.set_session_model(
                    session_id,
                    model_id,
                    toolsets=self._session_toolsets.get(session_id, []),
                    system_prompt=self._session_system_prompts.get(session_id),
                )
            except Exception as exc:  # noqa: BLE001
                log_event(
                    logger,
                    "acp.session.load.model_restore_failed",
                    level=logging.WARNING,
                    session_id=session_id,
                    model_id=model_id,
                    error=str(exc),
                )
                model_id = self._prompt_handler.model_id(session_id) or self._current_model_id()
        self._session_model_ids[session_id] = model_id
        history = self._load_session_history(session_id)
        self._session_history[session_id].extend(history)
        self._persist_session_meta(
            session_id,
            cwd_path,
            mcp_servers,
            current_mode=mode,
            current_model=model_id,
        )
        for note in history:
            await self._conn.session_update(session_id=note.session_id, update=note.update)  # type: ignore[arg-type]
        await self._send_available_commands(session_id)
        return LoadSessionResponse(config_options=self._session_config_options(session_id))

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

    async def fork_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: list[Any] | None = None,
        **_: Any,
    ) -> Any:
        """Forking is not implemented yet."""
        _ = (cwd, session_id, mcp_servers)
        raise RequestError.method_not_found("session/fork")

    async def resume_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: list[Any] | None = None,
        **_: Any,
    ) -> Any:
        """Resuming is not implemented yet."""
        _ = (cwd, session_id, mcp_servers)
        raise RequestError.method_not_found("session/resume")

    async def set_session_mode(self, mode_id: str, session_id: str, **_: Any) -> SetSessionModeResponse | None:
        """Update the current session mode and broadcast (Session Modes)."""
        with log_context(session_id=session_id, mode_id=mode_id):
            log_event(logger, "acp.session.mode")
        allowed_mode_ids = {item["id"] for item in self._available_modes()}
        if mode_id not in allowed_mode_ids:
            raise RequestError.invalid_params({"message": f"Unknown mode id: {mode_id}"})
        self._session_modes[session_id] = mode_id
        await self._send_update(self._mode_update(session_id, mode_id))
        await self._send_update(self._config_options_update(session_id))
        self._persist_session_meta(
            session_id,
            self._session_cwds.get(session_id, Path.cwd()),
            self._session_mcp_servers.get(session_id, []),
            current_mode=mode_id,
            current_model=self._session_model_ids.get(session_id, self._current_model_id()),
        )
        return SetSessionModeResponse()

    async def set_session_model(self, model_id: str, session_id: str, **_: Any) -> SetSessionModelResponse | None:
        """Switch the backing model for a session."""
        with log_context(session_id=session_id, model_id=model_id):
            log_event(logger, "acp.session.model")
        previous_model_id = self._session_model_ids.get(session_id, model_registry.current_model_id())
        try:
            await self._prompt_handler.set_session_model(
                session_id,
                model_id,
                toolsets=self._session_toolsets.get(session_id, []),
                system_prompt=self._session_system_prompts.get(session_id),
            )
            self._session_model_ids[session_id] = model_id
            try:
                model_registry.set_current_model(model_id)
            except Exception as exc:
                log_event(
                    logger,
                    "acp.session.model.persist_failed",
                    level=logging.WARNING,
                    session_id=session_id,
                    model_id=model_id,
                    error=str(exc),
                )
        except ModelBuildError as exc:
            try:
                model_registry.set_current_model(previous_model_id)
            except Exception as rollback_exc:
                log_event(
                    logger,
                    "acp.session.model.persist_rollback_failed",
                    level=logging.WARNING,
                    session_id=session_id,
                    model_id=previous_model_id,
                    error=str(rollback_exc),
                )
            payload = self._auth_required_from_model_build_error(exc)
            if payload:
                raise RequestError.auth_required(payload)
            raise
        await self._send_update(self._config_options_update(session_id))
        self._persist_session_meta(
            session_id,
            self._session_cwds.get(session_id, Path.cwd()),
            self._session_mcp_servers.get(session_id, []),
            current_mode=self._session_modes.get(session_id, "ask"),
            current_model=model_id,
        )
        return SetSessionModelResponse()

    def _auth_required_from_model_build_error(self, exc: ModelBuildError) -> dict[str, Any] | None:
        error_text = str(exc)
        env_names = {
            token.strip()
            for token in re.findall(r"\b[A-Z][A-Z0-9_]*\b", error_text)
            if token.strip().endswith(("_API_KEY", "_TOKEN"))
        }
        if not env_names:
            return None

        matched_methods = []
        for method in self._auth_methods:
            env_name = auth_method_env_var_name(method)
            if env_name and env_name in env_names:
                matched_methods.append(auth_method_payload(method))
        if not matched_methods:
            return None

        primary_missing = next(iter(sorted(env_names)))
        return {
            "authMethods": matched_methods,
            "missingEnvVar": primary_missing,
        }

    async def set_config_option(
        self,
        config_id: str,
        session_id: str,
        value: str,
        **_: Any,
    ) -> SetSessionConfigOptionResponse:
        """Set session config options (mode/model) via ACP session-config-options API."""
        with log_context(session_id=session_id, config_id=config_id, value=value):
            log_event(logger, "acp.session.config_option")
        if config_id == self.MODE_CONFIG_ID:
            await self.set_session_mode(mode_id=value, session_id=session_id)
            return SetSessionConfigOptionResponse(config_options=self._session_config_options(session_id))
        if config_id == self.MODEL_CONFIG_ID:
            await self.set_session_model(model_id=value, session_id=session_id)
            return SetSessionConfigOptionResponse(config_options=self._session_config_options(session_id))
        raise RequestError.invalid_params({"message": f"Unknown config option id: {config_id}"})

    def _mode_update(self, session_id: str, mode_id: str) -> SessionNotification:
        from acp.helpers import session_notification
        from acp.schema import CurrentModeUpdate

        return session_notification(
            session_id,
            CurrentModeUpdate(session_update="current_mode_update", current_mode_id=mode_id),
        )

    def _current_model_id(self) -> str:
        return model_registry.current_model_id()

    @staticmethod
    def _available_modes() -> list[dict[str, str]]:
        return available_modes()

    def _session_config_options(self, session_id: str) -> list[SessionConfigOption]:
        mode_id = self._session_modes.get(session_id, "ask")
        model_id = self._session_model_ids.get(session_id, self._current_model_id())
        mode_options = [
            SessionConfigSelectOption(
                name=entry["name"],
                value=entry["id"],
                description=entry.get("description"),
            )
            for entry in self._available_modes()
        ]
        model_options = [
            SessionConfigSelectOption(
                name=model_id_entry,
                value=model_id_entry,
                description=str(meta.get("description") or "") or None,
            )
            for model_id_entry, meta in model_registry.list_user_models().items()
        ]
        return [
            SessionConfigOption(
                root=SessionConfigOptionSelect(
                    id=self.MODE_CONFIG_ID,
                    name="Mode",
                    description="How the agent handles permission prompts.",
                    category="session",
                    type="select",
                    current_value=mode_id,
                    options=mode_options,
                )
            ),
            SessionConfigOption(
                root=SessionConfigOptionSelect(
                    id=self.MODEL_CONFIG_ID,
                    name="Model",
                    description="Model used for this session.",
                    category="session",
                    type="select",
                    current_value=model_id,
                    options=model_options,
                )
            ),
        ]

    def _config_options_update(self, session_id: str) -> SessionNotification:
        from acp.helpers import session_notification

        return session_notification(
            session_id,
            ConfigOptionUpdate(
                session_update="config_option_update",
                config_options=self._session_config_options(session_id),
            ),
        )

    def _persist_session_meta(
        self,
        session_id: str,
        cwd: Path,
        mcp_servers: list[Any] | None,
        *,
        current_mode: str,
        current_model: str,
    ) -> None:
        self._session_store.persist_meta(
            session_id,
            cwd,
            mcp_servers or [],
            current_mode=current_mode,
            current_model=current_model,
        )

    def _load_session_meta(self, session_id: str) -> dict[str, Any]:
        return self._session_store.load_meta(session_id)

    def _load_session_history(self, session_id: str) -> list[SessionNotification]:
        return self._session_store.load_history(session_id)

    @staticmethod
    def _require_absolute_cwd(cwd: str) -> Path:
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
