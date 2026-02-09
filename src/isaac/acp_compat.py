"""ACP SDK compatibility helpers for protocol features missing in wrappers.

The ACP 0.8.0 schema includes `session/set_config_option`, but current Python
SDK connection/router wrappers do not expose it yet. Isaac patches the runtime
router/connection surfaces so we can use the protocol method directly.
"""

from __future__ import annotations

from typing import Any

_PATCHED = False


def enable_session_config_options_api() -> None:
    """Enable `session/set_config_option` support on ACP SDK wrappers."""
    global _PATCHED
    if _PATCHED:
        return

    from acp.agent import connection as agent_connection_module
    from acp.agent import router as agent_router_module
    from acp.client import connection as client_connection_module
    from acp.meta import AGENT_METHODS
    from acp.schema import SetSessionConfigOptionRequest, SetSessionConfigOptionResponse
    from acp.utils import normalize_result, param_model, request_model_from_dict

    original_build_router = agent_router_module.build_agent_router

    if not getattr(original_build_router, "_isaac_config_option_patch", False):

        def _patched_build_agent_router(agent: Any, use_unstable_protocol: bool = False):
            router = original_build_router(agent, use_unstable_protocol=use_unstable_protocol)
            router.route_request(
                AGENT_METHODS["session_set_config_option"],
                SetSessionConfigOptionRequest,
                agent,
                "set_session_config_option",
                adapt_result=normalize_result,
            )
            return router

        setattr(_patched_build_agent_router, "_isaac_config_option_patch", True)
        agent_router_module.build_agent_router = _patched_build_agent_router
        agent_connection_module.build_agent_router = _patched_build_agent_router

    if not hasattr(client_connection_module.ClientSideConnection, "set_session_config_option"):

        @param_model(SetSessionConfigOptionRequest)
        async def _set_session_config_option(
            self: Any,
            config_id: str,
            session_id: str,
            value: str,
            **kwargs: Any,
        ) -> SetSessionConfigOptionResponse:
            return await request_model_from_dict(
                self._conn,
                AGENT_METHODS["session_set_config_option"],
                SetSessionConfigOptionRequest(
                    config_id=config_id,
                    session_id=session_id,
                    value=value,
                    field_meta=kwargs or None,
                ),
                SetSessionConfigOptionResponse,
            )

        setattr(client_connection_module.ClientSideConnection, "set_session_config_option", _set_session_config_option)

    _PATCHED = True
