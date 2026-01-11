"""OAuth token storage helpers."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CONFIG_DIR = Path(os.getenv("XDG_CONFIG_HOME") or (Path.home() / ".config")) / "isaac"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
TOKENS_FILE = CONFIG_DIR / "oauth_tokens.json"


@dataclass
class OAuthTokenSet:
    access_token: str
    refresh_token: str
    expires_at: float
    scope: str | None = None
    token_type: str | None = None
    account_id: str | None = None

    def is_expired(self, skew_s: int = 60) -> bool:
        return time.time() >= (self.expires_at - skew_s)

    def to_dict(self) -> dict[str, Any]:
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_at": self.expires_at,
            "scope": self.scope,
            "token_type": self.token_type,
            "account_id": self.account_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OAuthTokenSet":
        return cls(
            access_token=str(data.get("access_token") or ""),
            refresh_token=str(data.get("refresh_token") or ""),
            expires_at=float(data.get("expires_at") or 0),
            scope=data.get("scope"),
            token_type=data.get("token_type"),
            account_id=data.get("account_id"),
        )

    def expires_at_display(self) -> str:
        if not self.expires_at:
            return "unknown"
        when = datetime.fromtimestamp(self.expires_at, tz=timezone.utc)
        return when.isoformat()


def load_tokens() -> dict[str, Any]:
    if not TOKENS_FILE.exists():
        return {}
    try:
        return json.loads(TOKENS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_tokens(payload: dict[str, Any]) -> None:
    data = json.dumps(payload, indent=2, sort_keys=True)
    TOKENS_FILE.write_text(data, encoding="utf-8")
    try:
        os.chmod(TOKENS_FILE, 0o600)
    except Exception:
        pass


def load_provider_tokens(provider_key: str) -> OAuthTokenSet | None:
    data = load_tokens().get(provider_key)
    if not isinstance(data, dict):
        return None
    tokens = OAuthTokenSet.from_dict(data)
    if not tokens.access_token or not tokens.refresh_token:
        return None
    return tokens


def save_provider_tokens(provider_key: str, tokens: OAuthTokenSet) -> None:
    payload = load_tokens()
    payload[provider_key] = tokens.to_dict()
    save_tokens(payload)


def clear_provider_tokens(provider_key: str) -> None:
    payload = load_tokens()
    if provider_key in payload:
        payload.pop(provider_key)
        save_tokens(payload)
