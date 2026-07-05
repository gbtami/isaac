"""OpenAI Codex model discovery and configuration helpers."""

from __future__ import annotations

import base64
import json
import os
import platform
from dataclasses import dataclass
from typing import Any

import httpx

from isaac.agent.oauth.openai_codex.storage import OAuthTokenSet
from isaac.paths import config_dir

OPENAI_CODEX_BASE_URL = os.getenv("ISAAC_OPENAI_CODEX_BASE_URL", "https://chatgpt.com/backend-api/codex")
OPENAI_CODEX_ORIGINATOR = os.getenv("ISAAC_OPENAI_OAUTH_ORIGINATOR", "isaac")
OPENAI_CODEX_CLIENT_VERSION = os.getenv("ISAAC_OPENAI_OAUTH_CLIENT_VERSION", "0.72.0")

DEFAULT_CODEX_MODELS = [
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.4-mini",
]
# ChatGPT-login Codex models are not the same as OpenAI API models. Keep this
# allow-list deliberately small and update it with the Codex CLI/docs instead of
# trusting the generic models.dev OpenAI provider catalog. Spark is account-gated,
# so it is only useful when returned by live discovery, but accepting it here also
# lets a manual user config work for eligible accounts.
OPTIONAL_CODEX_MODELS = [
    "gpt-5.3-codex-spark",
]
OPENAI_CODEX_OAUTH_SOURCE = "openai-codex-oauth"

CONFIG_DIR = config_dir()
MODELS_FILE = CONFIG_DIR / "models.json"


def normalize_codex_model_name(model: object) -> str | None:
    """Return the ChatGPT-Codex model slug, or None for an invalid value."""

    name = str(model or "").strip()
    if not name:
        return None
    if name.startswith("openai-codex:"):
        name = name.split(":", 1)[1]
    if name.startswith("openai/"):
        name = name.split("/", 1)[1]
    return name.strip() or None


def is_supported_chatgpt_codex_model(model: object) -> bool:
    """Return whether a model should be offered for ChatGPT-login Codex OAuth."""

    name = normalize_codex_model_name(model)
    if not name:
        return False
    return name in {*DEFAULT_CODEX_MODELS, *OPTIONAL_CODEX_MODELS}


def _filter_chatgpt_codex_models(models: list[str]) -> list[str]:
    """Normalize, de-duplicate, and drop deprecated/API-only Codex models."""

    filtered: list[str] = []
    seen: set[str] = set()
    for raw_model in models:
        name = normalize_codex_model_name(raw_model)
        if not name or name in seen:
            continue
        if not is_supported_chatgpt_codex_model(name):
            continue
        filtered.append(name)
        seen.add(name)
    return filtered


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
        return list(DEFAULT_CODEX_MODELS), True

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
                filtered = _filter_chatgpt_codex_models(models)
                if filtered:
                    return filtered, False
    except Exception:
        pass

    return list(DEFAULT_CODEX_MODELS), True


def add_openai_codex_models(models: list[str]) -> int:
    """Upsert the live Codex OAuth model list and prune stale synced entries."""

    config = _load_models_config()
    models_config = config.setdefault("models", {})
    added = 0
    changed = False

    live_names = _filter_chatgpt_codex_models(models)
    live_ids = {f"openai-codex:{name}" for name in live_names}

    for model_id, meta in list(models_config.items()):
        if not isinstance(meta, dict):
            continue
        if str(meta.get("provider") or "").lower() != "openai-codex":
            continue
        if str(meta.get("oauth_source") or "").lower() != OPENAI_CODEX_OAUTH_SOURCE:
            continue
        if model_id not in live_ids:
            del models_config[model_id]
            changed = True

    for name in live_names:
        model_id = f"openai-codex:{name}"
        model_entry = {
            "provider": "openai-codex",
            "model": name,
            "description": f"OpenAI Codex {name} (OAuth, ChatGPT login)",
            "oauth_source": OPENAI_CODEX_OAUTH_SOURCE,
        }
        existing = models_config.get(model_id)
        if not isinstance(existing, dict):
            models_config[model_id] = model_entry
            added += 1
            changed = True
            continue
        merged = {**existing, **model_entry}
        if merged != existing:
            models_config[model_id] = merged
            changed = True

    if changed:
        _save_models_config(config)
    return added


async def sync_openai_codex_models(tokens: OAuthTokenSet) -> OpenAICodexModelSync:
    try:
        account_id = tokens.account_id or extract_account_id(None, tokens.access_token)
        if not account_id:
            models = list(DEFAULT_CODEX_MODELS)
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
