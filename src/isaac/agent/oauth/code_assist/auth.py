"""OAuth and onboarding helpers for Google Code Assist."""

from __future__ import annotations

import base64
import hashlib
import os
import secrets
import time
import webbrowser
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx

from isaac.agent.oauth.code_assist.client import CodeAssistClient
from isaac.agent.oauth.code_assist.endpoints import code_assist_load_endpoints
from isaac.agent.oauth.code_assist.storage import CodeAssistTokens, clear_tokens, load_tokens, save_tokens
from isaac.agent.oauth.callback_server import OAuthCallbackServer, OAuthCallbackServerConfig

GOOGLE_OAUTH_CLIENT_ID = os.getenv(
    "ISAAC_CODE_ASSIST_CLIENT_ID",
    "1071006060591-tmhssin2h21lcre235vtolojh4g403ep.apps.googleusercontent.com",
)
GOOGLE_OAUTH_CLIENT_SECRET = os.getenv("ISAAC_CODE_ASSIST_CLIENT_SECRET", "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")
GOOGLE_OAUTH_TIMEOUT_S = float(os.getenv("ISAAC_CODE_ASSIST_TIMEOUT_S", "300"))
GOOGLE_OAUTH_SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/cclog",
    "https://www.googleapis.com/auth/experimentsandconfigs",
]
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"


@dataclass(frozen=True)
class CodeAssistOAuthSession:
    auth_url: str
    redirect_uri: str
    pkce_verifier: str
    callback_server: OAuthCallbackServer
    timeout_s: float

    async def exchange_tokens(self) -> CodeAssistTokens:
        code = await self.callback_server.wait_for_code(self.timeout_s)
        return await exchange_code_for_tokens(code, self.redirect_uri, self.pkce_verifier)


def code_assist_status() -> str:
    tokens = load_tokens()
    if not tokens:
        return "Code Assist: not logged in"
    expiry = _format_expiry(tokens.expires_at, tokens.is_expired())
    suffix = f" ({expiry})"
    if tokens.is_expired():
        suffix = f" (access token expired, refresh on use; {expiry})"
    project_info = f" project={tokens.project_id}" if tokens.project_id else ""
    return f"Code Assist: logged in{suffix}{project_info}"


def _format_expiry(expires_at: float, expired: bool) -> str:
    if not expires_at:
        return "expires unknown"
    local = datetime.fromtimestamp(expires_at).astimezone().isoformat(timespec="seconds")
    delta = expires_at - time.time()
    if expired:
        return f"expired {local} local, {_format_duration(delta)} ago"
    return f"expires {local} local, in {_format_duration(delta)}"


def _format_duration(seconds: float) -> str:
    total = int(abs(seconds))
    if total < 60:
        return f"{total}s"
    minutes, sec = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {minutes}m" if minutes else f"{hours}h"
    days, hours = divmod(hours, 24)
    return f"{days}d {hours}h" if hours else f"{days}d"


def clear_code_assist_tokens() -> None:
    clear_tokens()


def maybe_open_browser(url: str) -> bool:
    if os.getenv("ISAAC_NO_BROWSER"):
        return False
    try:
        return webbrowser.open(url)
    except Exception:
        return False


async def begin_code_assist_oauth() -> CodeAssistOAuthSession:
    pkce_verifier, pkce_challenge = _generate_pkce()
    state = _generate_state()
    callback = OAuthCallbackServer(
        OAuthCallbackServerConfig(
            path="/oauth2callback",
            state=state,
            success_html=_HTML_SUCCESS,
            error_html=_html_error,
            listen_host=_get_listen_host(),
            redirect_host="localhost",
        )
    )
    redirect_uri = callback.start()
    auth_url = _build_authorize_url(redirect_uri, pkce_challenge, state)
    return CodeAssistOAuthSession(
        auth_url=auth_url,
        redirect_uri=redirect_uri,
        pkce_verifier=pkce_verifier,
        callback_server=callback,
        timeout_s=GOOGLE_OAUTH_TIMEOUT_S,
    )


async def finalize_code_assist_login(tokens: CodeAssistTokens) -> CodeAssistTokens:
    client = CodeAssistClient(base_urls=code_assist_load_endpoints())
    project_id, user_tier = await setup_user(client, tokens.access_token)
    tokens.project_id = project_id
    tokens.user_tier = user_tier
    save_tokens(tokens)
    return tokens


async def get_access_token() -> CodeAssistTokens:
    tokens = load_tokens()
    if not tokens:
        raise RuntimeError("Code Assist tokens not found. Run /login code-assist first.")
    if not tokens.is_expired():
        return tokens

    refreshed = await refresh_tokens(tokens.refresh_token)
    refreshed.project_id = tokens.project_id
    refreshed.user_tier = tokens.user_tier
    save_tokens(refreshed)
    return refreshed


async def ensure_project(tokens: CodeAssistTokens) -> CodeAssistTokens:
    if tokens.project_id:
        return tokens
    client = CodeAssistClient()
    project_id, user_tier = await setup_user(client, tokens.access_token)
    tokens.project_id = project_id
    tokens.user_tier = user_tier
    save_tokens(tokens)
    return tokens


async def exchange_code_for_tokens(code: str, redirect_uri: str, pkce_verifier: str) -> CodeAssistTokens:
    payload = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": GOOGLE_OAUTH_CLIENT_ID,
        "client_secret": GOOGLE_OAUTH_CLIENT_SECRET,
        "code_verifier": pkce_verifier,
    }
    data = await _post_form(GOOGLE_TOKEN_URL, payload)
    return _tokens_from_response(data)


async def refresh_tokens(refresh_token: str) -> CodeAssistTokens:
    payload = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": GOOGLE_OAUTH_CLIENT_ID,
        "client_secret": GOOGLE_OAUTH_CLIENT_SECRET,
    }
    try:
        data = await _post_form(GOOGLE_TOKEN_URL, payload)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 401:
            clear_tokens()
            raise RuntimeError(
                "Code Assist OAuth refresh failed (401 Unauthorized). Run /login code-assist to reauthenticate."
            ) from exc
        raise
    return _tokens_from_response(data, refresh_token=refresh_token)


async def setup_user(client: CodeAssistClient, access_token: str) -> tuple[str, str]:
    project_id = _project_env()
    metadata = _client_metadata(project_id)
    load_req = {"cloudaicompanionProject": project_id, "metadata": metadata}
    load = await client.post_method("loadCodeAssist", load_req, access_token)

    current_tier = load.get("currentTier")
    if current_tier:
        tier_id = current_tier.get("id") or "unknown"
        project = load.get("cloudaicompanionProject")
        if project:
            return project, tier_id
        if project_id:
            return project_id, tier_id
        raise RuntimeError(
            "This account requires GOOGLE_CLOUD_PROJECT or GOOGLE_CLOUD_PROJECT_ID to be set for Code Assist."
        )

    tier = _default_tier(load.get("allowedTiers") or [])
    tier_id = tier.get("id") or "unknown"
    if tier_id == "free-tier":
        onboard_req = {"tierId": tier_id, "cloudaicompanionProject": None, "metadata": metadata}
    else:
        onboard_req = {
            "tierId": tier_id,
            "cloudaicompanionProject": project_id,
            "metadata": _client_metadata(project_id, include_duet=True),
        }

    lro = await client.post_method("onboardUser", onboard_req, access_token)
    while not lro.get("done") and lro.get("name"):
        lro = await client.get_operation(lro["name"], access_token)

    response = (lro.get("response") or {}).get("cloudaicompanionProject") or {}
    if response.get("id"):
        return response["id"], tier_id
    if project_id:
        return project_id, tier_id
    raise RuntimeError("Code Assist onboarding did not return a project id.")


def _default_tier(allowed: list[dict[str, Any]]) -> dict[str, Any]:
    for tier in allowed:
        if tier.get("isDefault"):
            return tier
    return {
        "id": "legacy-tier",
        "userDefinedCloudaicompanionProject": True,
    }


def _client_metadata(project_id: str | None, include_duet: bool = True) -> dict[str, Any]:
    metadata = {
        "ideType": "IDE_UNSPECIFIED",
        "platform": "PLATFORM_UNSPECIFIED",
        "pluginType": "GEMINI",
    }
    if include_duet and project_id:
        metadata["duetProject"] = project_id
    return metadata


def _project_env() -> str | None:
    return os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT_ID")


def _build_authorize_url(redirect_uri: str, pkce_challenge: str, state: str) -> str:
    params = {
        "response_type": "code",
        "client_id": GOOGLE_OAUTH_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "scope": " ".join(GOOGLE_OAUTH_SCOPES),
        "access_type": "offline",
        "prompt": "consent",
        "code_challenge": pkce_challenge,
        "code_challenge_method": "S256",
        "state": state,
    }
    return f"{GOOGLE_AUTH_URL}?{httpx.QueryParams(params)}"


def _generate_pkce() -> tuple[str, str]:
    verifier = _random_string(64)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = _base64_url_encode(digest)
    return verifier, challenge


def _generate_state() -> str:
    return _base64_url_encode(secrets.token_bytes(32))


def _random_string(length: int) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"
    return "".join(secrets.choice(alphabet) for _ in range(length))


def _base64_url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


async def _post_form(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.post(url, data=payload, headers=headers)
        response.raise_for_status()
        return response.json()


def _tokens_from_response(data: dict[str, Any], refresh_token: str | None = None) -> CodeAssistTokens:
    access_token = str(data.get("access_token") or "")
    refresh = str(data.get("refresh_token") or refresh_token or "")
    expires_in = int(data.get("expires_in") or 3600)
    expires_at = time.time() + expires_in
    if not access_token or not refresh:
        raise RuntimeError("OAuth token response missing access or refresh token.")
    return CodeAssistTokens(
        access_token=access_token,
        refresh_token=refresh,
        expires_at=expires_at,
        scope=data.get("scope"),
        token_type=data.get("token_type"),
    )


def _get_listen_host() -> str:
    return os.getenv("ISAAC_OAUTH_CALLBACK_HOST", "localhost")


_HTML_SUCCESS = """<!doctype html>
<html>
  <head>
    <title>Isaac - Code Assist Authorization Successful</title>
    <style>
      body {
        font-family: system-ui, -apple-system, sans-serif;
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
        margin: 0;
        background: #101317;
        color: #e6edf3;
      }
      .container { text-align: center; padding: 2rem; }
      h1 { margin-bottom: 1rem; }
      p { color: #9aa7b1; }
    </style>
  </head>
  <body>
    <div class="container">
      <h1>Authorization Successful</h1>
      <p>You can close this window and return to Isaac.</p>
    </div>
    <script>setTimeout(() => window.close(), 2000)</script>
  </body>
</html>
"""


def _html_error(error: str) -> str:
    return f"""<!doctype html>
<html>
  <head>
    <title>Isaac - Code Assist Authorization Failed</title>
    <style>
      body {{
        font-family: system-ui, -apple-system, sans-serif;
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
        margin: 0;
        background: #101317;
        color: #e6edf3;
      }}
      .container {{ text-align: center; padding: 2rem; }}
      h1 {{ margin-bottom: 1rem; color: #ff6a5a; }}
      p {{ color: #9aa7b1; }}
      .error {{ margin-top: 1rem; font-family: monospace; color: #ffd4cc; }}
    </style>
  </head>
  <body>
    <div class="container">
      <h1>Authorization Failed</h1>
      <p>An error occurred during authorization.</p>
      <div class="error">{error}</div>
    </div>
  </body>
</html>
"""
