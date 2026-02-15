"""Storage helpers for Code Assist OAuth tokens."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from isaac.paths import config_dir

CONFIG_DIR = config_dir()
TOKENS_FILE = CONFIG_DIR / "code_assist.json"


@dataclass
class CodeAssistTokens:
    access_token: str
    refresh_token: str
    expires_at: float
    project_id: str | None = None
    user_tier: str | None = None
    scope: str | None = None
    token_type: str | None = None

    def is_expired(self, skew_s: int = 60) -> bool:
        return time.time() >= (self.expires_at - skew_s)

    def expires_at_display(self) -> str:
        if not self.expires_at:
            return "unknown"
        when = datetime.fromtimestamp(self.expires_at, tz=timezone.utc)
        return when.isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_at": self.expires_at,
            "project_id": self.project_id,
            "user_tier": self.user_tier,
            "scope": self.scope,
            "token_type": self.token_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CodeAssistTokens":
        return cls(
            access_token=str(data.get("access_token") or ""),
            refresh_token=str(data.get("refresh_token") or ""),
            expires_at=float(data.get("expires_at") or 0),
            project_id=data.get("project_id"),
            user_tier=data.get("user_tier"),
            scope=data.get("scope"),
            token_type=data.get("token_type"),
        )


def load_tokens() -> CodeAssistTokens | None:
    if not TOKENS_FILE.exists():
        return None
    try:
        data = json.loads(TOKENS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    tokens = CodeAssistTokens.from_dict(data)
    if not tokens.access_token or not tokens.refresh_token:
        return None
    return tokens


def save_tokens(tokens: CodeAssistTokens) -> None:
    payload = json.dumps(tokens.to_dict(), indent=2, sort_keys=True)
    TOKENS_FILE.write_text(payload, encoding="utf-8")
    try:
        os.chmod(TOKENS_FILE, 0o600)
    except Exception:
        pass


def clear_tokens() -> None:
    try:
        TOKENS_FILE.unlink(missing_ok=True)
    except Exception:
        pass
