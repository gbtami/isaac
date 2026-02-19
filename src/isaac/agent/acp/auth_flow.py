"""ACP authenticate flow helpers."""

from __future__ import annotations

import logging
import os
import sys

from acp import AuthenticateResponse, RequestError
from acp.schema import AuthMethod

from isaac.agent.acp.auth_methods import auth_method_env_var_name, auth_method_payload, auth_method_type
from isaac.agent.oauth.code_assist import begin_code_assist_oauth, finalize_code_assist_login
from isaac.agent.oauth.code_assist.auth import maybe_open_browser as maybe_open_code_assist
from isaac.agent.oauth.code_assist.storage import load_tokens as load_code_assist_tokens
from isaac.agent.oauth.openai_codex import begin_openai_oauth, has_openai_tokens, save_openai_tokens
from isaac.agent.oauth.openai_codex.auth import maybe_open_browser as maybe_open_openai
from isaac.agent.oauth.openai_codex.model import sync_openai_codex_models
from isaac.log_utils import log_event

logger = logging.getLogger(__name__)


def _emit_oauth_notice(provider: str, auth_url: str, opened: bool) -> None:
    notice = (
        f"{provider} OAuth login started. Browser opened automatically.\n"
        if opened
        else f"{provider} OAuth login started. Open this URL in your browser:\n"
    )
    print(f"{notice}{auth_url}", file=sys.stderr, flush=True)


def has_code_assist_tokens() -> bool:
    return load_code_assist_tokens() is not None


async def authenticate_method(method: AuthMethod, *, method_id: str) -> AuthenticateResponse:
    """Handle ACP authenticate for a single advertised auth method."""
    method_type = auth_method_type(method)

    if method_type == "env_var":
        from isaac.agent.models import load_runtime_env

        env_name = auth_method_env_var_name(method)
        if not env_name:
            raise RequestError.invalid_params({"message": f"Auth method {method_id} is missing varName."})
        load_runtime_env()
        if not os.getenv(env_name):
            raise RequestError.auth_required({"authMethods": [auth_method_payload(method)], "missingEnvVar": env_name})
        return AuthenticateResponse()

    if method_id == "openai":
        if not has_openai_tokens():
            session = await begin_openai_oauth()
            opened = maybe_open_openai(session.auth_url)
            _emit_oauth_notice("OpenAI Codex", session.auth_url, opened)
            tokens = await session.exchange_tokens()
            save_openai_tokens(tokens)
            sync = await sync_openai_codex_models(tokens)
            log_event(
                logger,
                "acp.authenticate.openai.sync",
                added=sync.added,
                total=sync.total,
                used_fallback=sync.used_fallback,
                error=sync.error or "",
            )
        return AuthenticateResponse()

    if method_id == "code-assist":
        if not has_code_assist_tokens():
            session = await begin_code_assist_oauth()
            opened = maybe_open_code_assist(session.auth_url)
            _emit_oauth_notice("Code Assist", session.auth_url, opened)
            tokens = await session.exchange_tokens()
            await finalize_code_assist_login(tokens)
        return AuthenticateResponse()

    # Custom/unknown agent-auth method: treat as no-op success after method id validation.
    return AuthenticateResponse()
