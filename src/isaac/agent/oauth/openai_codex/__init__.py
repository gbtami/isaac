"""OpenAI Codex OAuth helper."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import os
import platform
import secrets
import time
import webbrowser
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx

from isaac.agent.oauth.callback_server import OAuthCallbackServer, OAuthCallbackServerConfig
from isaac.agent.oauth.openai_codex.models import extract_account_id
from isaac.agent.oauth.storage import OAuthTokenSet, clear_provider_tokens, load_provider_tokens, save_provider_tokens

OPENAI_CODEX_PROVIDER_KEY = "openai-codex"
OPENAI_CODEX_BASE_URL = os.getenv("ISAAC_OPENAI_CODEX_BASE_URL", "https://chatgpt.com/backend-api/codex")
OPENAI_OAUTH_CLIENT_ID = os.getenv("ISAAC_OPENAI_OAUTH_CLIENT_ID", "app_EMoamEEZ73f0CkXaXp7hrann")
OPENAI_OAUTH_ISSUER = os.getenv("ISAAC_OPENAI_OAUTH_ISSUER", "https://auth.openai.com")
OPENAI_OAUTH_TIMEOUT_S = float(os.getenv("ISAAC_OPENAI_OAUTH_TIMEOUT_S", "120"))
OPENAI_OAUTH_SCOPE = "openid profile email offline_access"
OPENAI_OAUTH_ORIGINATOR = os.getenv("ISAAC_OPENAI_OAUTH_ORIGINATOR", "codex_cli_rs")
OPENAI_OAUTH_CLIENT_VERSION = os.getenv("ISAAC_OPENAI_OAUTH_CLIENT_VERSION", "0.72.0")

_OPENAI_TOKEN_LOCK = asyncio.Lock()


@dataclass(frozen=True)
class OpenAIOAuthSession:
    auth_url: str
    redirect_uri: str
    pkce_verifier: str
    callback_server: OAuthCallbackServer
    timeout_s: float

    async def exchange_tokens(self) -> OAuthTokenSet:
        code = await self.callback_server.wait_for_code(self.timeout_s)
        return await exchange_code_for_tokens(code, self.redirect_uri, self.pkce_verifier)


def openai_token_status() -> str:
    tokens = load_provider_tokens(OPENAI_CODEX_PROVIDER_KEY)
    if not tokens:
        return "OpenAI Codex: not logged in"
    expiry = _format_expiry(tokens.expires_at, tokens.is_expired())
    suffix = f" ({expiry})"
    if tokens.is_expired():
        suffix = f" (access token expired, refresh on use; {expiry})"
    account = f" account={tokens.account_id}" if tokens.account_id else ""
    return f"OpenAI Codex: logged in{suffix}{account}"


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


def save_openai_tokens(tokens: OAuthTokenSet) -> None:
    save_provider_tokens(OPENAI_CODEX_PROVIDER_KEY, tokens)


def clear_openai_tokens() -> None:
    clear_provider_tokens(OPENAI_CODEX_PROVIDER_KEY)


def has_openai_tokens() -> bool:
    return load_provider_tokens(OPENAI_CODEX_PROVIDER_KEY) is not None


async def begin_openai_oauth() -> OpenAIOAuthSession:
    pkce_verifier, pkce_challenge = _generate_pkce()
    state = _generate_state()
    callback = OAuthCallbackServer(
        OAuthCallbackServerConfig(
            path="/auth/callback",
            state=state,
            success_html=_HTML_SUCCESS,
            error_html=_html_error,
            listen_host=_get_listen_host(),
            redirect_host="localhost",
            port=_get_listen_port(),
        )
    )
    redirect_uri = callback.start()
    auth_url = _build_authorize_url(redirect_uri, pkce_challenge, state)
    return OpenAIOAuthSession(
        auth_url=auth_url,
        redirect_uri=redirect_uri,
        pkce_verifier=pkce_verifier,
        callback_server=callback,
        timeout_s=OPENAI_OAUTH_TIMEOUT_S,
    )


async def openai_auth_request_hook(request: httpx.Request) -> None:
    tokens = await get_openai_tokens()
    request.headers["authorization"] = f"Bearer {tokens.access_token}"
    if tokens.account_id:
        request.headers.setdefault("ChatGPT-Account-Id", tokens.account_id)
    request.headers.setdefault("originator", OPENAI_OAUTH_ORIGINATOR)
    request.headers.setdefault("User-Agent", _build_codex_user_agent())


async def get_access_token() -> str:
    return (await get_openai_tokens()).access_token


async def get_openai_tokens() -> OAuthTokenSet:
    tokens = load_provider_tokens(OPENAI_CODEX_PROVIDER_KEY)
    if not tokens:
        raise RuntimeError("OpenAI Codex OAuth tokens not found. Run /login openai first.")
    if not tokens.is_expired():
        if not tokens.account_id:
            account_id = extract_account_id(None, tokens.access_token)
            if account_id:
                tokens.account_id = account_id
                save_openai_tokens(tokens)
        return tokens

    async with _OPENAI_TOKEN_LOCK:
        tokens = load_provider_tokens(OPENAI_CODEX_PROVIDER_KEY)
        if not tokens:
            raise RuntimeError("OpenAI Codex OAuth tokens not found. Run /login openai first.")
        if tokens.is_expired():
            tokens = await refresh_tokens(tokens.refresh_token, account_id=tokens.account_id)
            save_openai_tokens(tokens)
        if not tokens.account_id:
            account_id = extract_account_id(None, tokens.access_token)
            if account_id:
                tokens.account_id = account_id
                save_openai_tokens(tokens)
        return tokens


def maybe_open_browser(url: str) -> bool:
    if os.getenv("ISAAC_NO_BROWSER"):
        return False
    try:
        return webbrowser.open(url)
    except Exception:
        return False


async def exchange_code_for_tokens(code: str, redirect_uri: str, pkce_verifier: str) -> OAuthTokenSet:
    payload = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": OPENAI_OAUTH_CLIENT_ID,
        "code_verifier": pkce_verifier,
    }
    data = await _post_form(f"{OPENAI_OAUTH_ISSUER}/oauth/token", payload)
    return _tokens_from_response(data)


async def refresh_tokens(refresh_token: str, *, account_id: str | None = None) -> OAuthTokenSet:
    payload = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": OPENAI_OAUTH_CLIENT_ID,
    }
    data = await _post_form(f"{OPENAI_OAUTH_ISSUER}/oauth/token", payload)
    return _tokens_from_response(data, refresh_token=refresh_token, account_id=account_id)


def _tokens_from_response(
    data: dict[str, Any],
    refresh_token: str | None = None,
    account_id: str | None = None,
) -> OAuthTokenSet:
    access_token = str(data.get("access_token") or "")
    refresh = str(data.get("refresh_token") or refresh_token or "")
    expires_in = int(data.get("expires_in") or 3600)
    expires_at = time.time() + expires_in
    if not access_token or not refresh:
        raise RuntimeError("OAuth token response missing access or refresh token.")
    id_token = str(data.get("id_token") or "")
    resolved_account_id = extract_account_id(id_token, access_token) or account_id
    return OAuthTokenSet(
        access_token=access_token,
        refresh_token=refresh,
        expires_at=expires_at,
        scope=data.get("scope"),
        token_type=data.get("token_type"),
        account_id=resolved_account_id,
    )


def _build_authorize_url(redirect_uri: str, pkce_challenge: str, state: str) -> str:
    params = {
        "response_type": "code",
        "client_id": OPENAI_OAUTH_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "scope": OPENAI_OAUTH_SCOPE,
        "code_challenge": pkce_challenge,
        "code_challenge_method": "S256",
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
        "state": state,
    }
    return f"{OPENAI_OAUTH_ISSUER}/oauth/authorize?{httpx.QueryParams(params)}"


def _build_codex_user_agent() -> str:
    os_name = platform.system()
    if os_name == "Darwin":
        os_name = "Mac OS"
    os_version = platform.release()
    arch = platform.machine()
    return (
        f"{OPENAI_OAUTH_ORIGINATOR}/{OPENAI_OAUTH_CLIENT_VERSION} ({os_name} {os_version}; {arch}) Terminal_Codex_CLI"
    )


def _generate_pkce() -> tuple[str, str]:
    verifier = _random_string(43)
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


def _get_listen_host() -> str:
    return os.getenv("ISAAC_OAUTH_CALLBACK_HOST", "localhost")


def _get_listen_port() -> int:
    port_raw = os.getenv("ISAAC_OPENAI_OAUTH_PORT")
    if not port_raw:
        return 1455
    try:
        port = int(port_raw)
    except ValueError:
        return 1455
    return port if port > 0 else 0


_HTML_SUCCESS = """<!doctype html>
<html>
  <head>
    <title>Isaac - OpenAI Authorization Successful</title>
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
    <title>Isaac - OpenAI Authorization Failed</title>
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
