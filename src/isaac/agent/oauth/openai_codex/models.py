"""OpenAI Codex model discovery and configuration helpers."""

from __future__ import annotations

import base64
import json
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from isaac.agent.oauth.storage import OAuthTokenSet

OPENAI_CODEX_BASE_URL = os.getenv("ISAAC_OPENAI_CODEX_BASE_URL", "https://chatgpt.com/backend-api/codex")
OPENAI_CODEX_ORIGINATOR = os.getenv("ISAAC_OPENAI_OAUTH_ORIGINATOR", "isaac")
OPENAI_CODEX_CLIENT_VERSION = os.getenv("ISAAC_OPENAI_OAUTH_CLIENT_VERSION", "0.72.0")

DEFAULT_CODEX_MODELS = [
    "gpt-5.2",
    "gpt-5.2-codex",
]

CONFIG_DIR = Path(os.getenv("XDG_CONFIG_HOME") or (Path.home() / ".config")) / "isaac"
MODELS_FILE = CONFIG_DIR / "models.json"


@dataclass(frozen=True)
class OpenAICodexModelSync:
    added: int
    total: int
    used_fallback: bool
    error: str | None = None


def parse_jwt_claims(token: str) -> dict[str, Any] | None:
    if not token or token.count(".") != 2:
        return None
    try:
        _header, payload, _sig = token.split(".")
        padded = payload + "=" * (-len(payload) % 4)
        data = base64.urlsafe_b64decode(padded.encode("ascii"))
        return json.loads(data.decode("utf-8"))
    except Exception:
        return None


def extract_account_id(id_token: str | None, access_token: str | None) -> str | None:
    for token in (id_token, access_token):
        claims = parse_jwt_claims(token or "")
        if not claims:
            continue
        auth_claims = claims.get("https://api.openai.com/auth")
        if isinstance(auth_claims, dict):
            account_id = auth_claims.get("chatgpt_account_id")
            if account_id:
                return str(account_id)
    return None


def _build_codex_user_agent(originator: str, client_version: str) -> str:
    os_name = platform.system()
    if os_name == "Darwin":
        os_name = "Mac OS"
    os_version = platform.release()
    arch = platform.machine()
    return f"{originator}/{client_version} ({os_name} {os_version}; {arch}) Terminal_Codex_CLI"


async def fetch_codex_models(
    access_token: str,
    account_id: str,
    *,
    base_url: str = OPENAI_CODEX_BASE_URL,
    originator: str = OPENAI_CODEX_ORIGINATOR,
    client_version: str = OPENAI_CODEX_CLIENT_VERSION,
) -> tuple[list[str], bool]:
    if not access_token or not account_id:
        return DEFAULT_CODEX_MODELS, True

    models_url = f"{base_url.rstrip('/')}/models"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "ChatGPT-Account-Id": account_id,
        "User-Agent": _build_codex_user_agent(originator, client_version),
        "originator": originator,
        "Accept": "application/json",
    }
    params = {"client_version": client_version}

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(models_url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, dict) and isinstance(data.get("models"), list):
                models: list[str] = []
                for entry in data["models"]:
                    if not isinstance(entry, dict):
                        continue
                    model_id = entry.get("slug") or entry.get("id") or entry.get("name")
                    if model_id:
                        models.append(str(model_id))
                if models:
                    return list(dict.fromkeys(models)), False
    except Exception:
        pass

    return DEFAULT_CODEX_MODELS, True


def add_openai_codex_models(models: list[str]) -> int:
    config = _load_models_config()
    models_config = config.setdefault("models", {})
    added = 0

    for model_name in models:
        name = str(model_name).strip()
        if not name:
            continue
        model_id = f"openai-codex:{name}"
        if model_id in models_config:
            continue
        models_config[model_id] = {
            "provider": "openai-codex",
            "model": name,
            "description": f"OpenAI Codex {name} (OAuth, ChatGPT login)",
            "oauth_source": "openai-codex-oauth",
        }
        added += 1

    if added:
        _save_models_config(config)
    return added


async def sync_openai_codex_models(tokens: OAuthTokenSet) -> OpenAICodexModelSync:
    try:
        account_id = tokens.account_id or extract_account_id(None, tokens.access_token)
        if not account_id:
            models = DEFAULT_CODEX_MODELS
            used_fallback = True
        else:
            models, used_fallback = await fetch_codex_models(
                tokens.access_token,
                account_id,
            )
        added = add_openai_codex_models(models)
        return OpenAICodexModelSync(added=added, total=len(models), used_fallback=used_fallback)
    except Exception as exc:  # noqa: BLE001
        return OpenAICodexModelSync(added=0, total=0, used_fallback=False, error=str(exc))


def _load_models_config() -> dict[str, Any]:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if MODELS_FILE.exists():
        try:
            return json.loads(MODELS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {"models": {}}
    return {"models": {}}


def _save_models_config(config: dict[str, Any]) -> None:
    MODELS_FILE.write_text(json.dumps(config, indent=2), encoding="utf-8")
