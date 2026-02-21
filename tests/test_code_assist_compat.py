from __future__ import annotations

from isaac.agent.oauth.code_assist.client import (
    CodeAssistClient,
    _header_styles,
)
from isaac.agent.oauth.code_assist.request import apply_code_assist_envelope


def test_code_assist_request_defaults_to_gemini_cli_shape(monkeypatch) -> None:
    payload = {
        "model": "gemini-2.5-pro",
        "request": {
            "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
            "systemInstruction": {"role": "system", "parts": [{"text": "stay concise"}]},
        },
    }
    shaped = apply_code_assist_envelope(payload, "gemini-2.5-pro", "projects/test")
    assert shaped["project"] == "projects/test"
    assert "requestType" not in shaped
    assert "requestId" not in shaped
    assert "userAgent" not in shaped
    assert "session_id" in shaped["request"]
    assert "role" not in shaped["request"]["systemInstruction"]


def test_code_assist_header_styles_default_to_gemini_cli(monkeypatch) -> None:
    monkeypatch.delenv("ISAAC_CODE_ASSIST_HEADER_STYLES", raising=False)
    assert _header_styles() == ["gemini-cli"]


def test_gemini_cli_headers_use_gemini_cli_user_agent() -> None:
    client = CodeAssistClient.__new__(CodeAssistClient)
    headers = client._headers("access-token", model_name="gemini-2.5-pro")
    assert headers["Authorization"] == "Bearer access-token"
    assert headers["User-Agent"].startswith("GeminiCLI/")
    assert "/gemini-2.5-pro " in headers["User-Agent"]
    assert "X-Goog-Api-Client" not in headers
    assert "Client-Metadata" not in headers
