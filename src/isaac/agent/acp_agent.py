"""Core ACP-compliant agent implementation."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict

from acp import Agent
from acp.agent.connection import AgentSideConnection

from isaac.agent.acp.extensions import ExtensionsMixin
from isaac.agent.acp.filesystem import FileSystemMixin
from isaac.agent.acp.initialization import InitializationMixin
from isaac.agent.acp.permissions import PermissionMixin
from isaac.agent.acp.prompts import PromptMixin
from isaac.agent.acp.sessions import SessionLifecycleMixin
from isaac.agent.acp.terminal import TerminalMixin
from isaac.agent.acp.tools import ToolCallsMixin
from isaac.agent.acp.updates import SessionUpdateMixin
from isaac.agent.agent_terminal import TerminalState
from isaac.agent.brain.prompt_handler import PromptHandler
from isaac.agent.brain.session_ops import RunnerFactory
from isaac.agent.constants import TOOL_OUTPUT_LIMIT
from isaac.agent.session_store import SessionStore
from isaac.log_utils import log_event

logger = logging.getLogger(__name__)
DEFAULT_COMMAND_TIMEOUT_S = 30.0


class ACPAgent(
    InitializationMixin,
    PermissionMixin,
    SessionLifecycleMixin,
    PromptMixin,
    ToolCallsMixin,
    FileSystemMixin,
    TerminalMixin,
    SessionUpdateMixin,
    ExtensionsMixin,
    Agent,
):
    """Implements ACP session, prompt, tool, filesystem, and terminal flows."""

    def __init__(
        self,
        conn: AgentSideConnection | None = None,
        *,
        agent_name: str = "isaac",
        agent_title: str = "Isaac ACP Agent",
        agent_version: str = "0.2.0",
        runner_factory: RunnerFactory | None = None,
    ) -> None:
        self._conn: AgentSideConnection | None = conn
        self._sessions: set[str] = set()
        self._session_cwds: dict[str, Path] = {}
        self._terminals: Dict[str, TerminalState] = {}
        self._session_modes: Dict[str, str] = {}
        self._cancel_events: Dict[str, asyncio.Event] = {}
        self._session_model_ids: Dict[str, str] = {}
        self._session_history: Dict[str, list[Any]] = {}
        self._session_allowed_commands: Dict[str, set[tuple[str, str]]] = {}
        self._session_mcp_servers: Dict[str, Any] = {}
        self._session_usage: Dict[str, Any] = {}
        self._session_toolsets: Dict[str, list[Any]] = {}
        self._agent_name = agent_name
        self._agent_title = agent_title
        self._agent_version = agent_version
        self._client_capabilities: Any | None = None
        self._client_info: Any | None = None
        self._session_commands_advertised: set[str] = set()
        self._command_timeout_s = DEFAULT_COMMAND_TIMEOUT_S
        self._tool_output_limit = TOOL_OUTPUT_LIMIT
        self._terminal_output_limit = TOOL_OUTPUT_LIMIT
        self._session_store = SessionStore(Path.home() / ".isaac" / "sessions")
        self._session_last_chunk: Dict[str, str | None] = {}
        self._runner_factory = runner_factory
        self._prompt_handler: PromptHandler = self._build_prompt_handler()
        self._session_system_prompts: Dict[str, str | None] = {}

    def on_connect(self, conn: AgentSideConnection) -> None:  # type: ignore[override]
        """Capture connection when wiring via run_agent/connect_to_agent."""
        self._conn = conn
        log_event(logger, "acp.connection.ready")
