"""Initialization and auth handlers for the ACP agent."""

from __future__ import annotations

import logging
from typing import Any

from acp import AuthenticateResponse, InitializeResponse, PROTOCOL_VERSION
from acp.schema import (
    AgentCapabilities,
    Implementation,
    McpCapabilities,
    PromptCapabilities,
    SessionCapabilities,
    SessionListCapabilities,
)
from isaac.log_utils import log_event

logger = logging.getLogger(__name__)


class InitializationMixin:
    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: Any | None = None,
        client_info: Any | None = None,
        **_: Any,
    ) -> InitializeResponse:
        """Handle ACP initialize handshake (Initialization section)."""
        log_event(logger, "acp.initialize.request", protocol_version=protocol_version)
        # ACP version negotiation: if the requested version isn't supported, respond with
        # the latest version we support; the client may disconnect if it can't accept it.
        if protocol_version != PROTOCOL_VERSION:
            log_event(
                logger,
                "acp.initialize.version_mismatch",
                level=logging.WARNING,
                requested=protocol_version,
                supported=PROTOCOL_VERSION,
            )
        # Capture peer capabilities/info for optional behavior gating.
        self._client_capabilities = client_capabilities
        self._client_info = client_info
        capabilities = AgentCapabilities(
            load_session=True,
            prompt_capabilities=PromptCapabilities(
                embedded_context=True,
                image=False,
                audio=False,
            ),
            mcp_capabilities=McpCapabilities(http=True, sse=True),
            session_capabilities=SessionCapabilities(list=SessionListCapabilities()),
        )
        capabilities.field_meta = {"extMethods": ["model/list", "model/set"]}

        return InitializeResponse(
            protocol_version=PROTOCOL_VERSION,
            agent_capabilities=capabilities,
            agent_info=Implementation(
                name=self._agent_name,
                title=self._agent_title,
                version=self._agent_version,
            ),
        )

    async def authenticate(self, method_id: str, **_: Any) -> AuthenticateResponse | None:
        """Return a no-op authentication response (Initialization auth step)."""
        log_event(logger, "acp.authenticate.request", method_id=method_id)
        return AuthenticateResponse()
