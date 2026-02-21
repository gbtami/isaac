from __future__ import annotations

import sys
from unittest.mock import AsyncMock

import pytest

from isaac.client.auth import auth_method_env_var_name, auth_method_link, auth_method_type, find_auth_method
from isaac.client.client import _auth_methods_for_error, _authenticate_if_needed


def test_auth_method_helpers_read_env_var_metadata() -> None:
    method = {
        "id": "env_var:openrouter_api_key",
        "name": "OpenRouter API key",
        "_meta": {
            "type": "env_var",
            "varName": "OPENROUTER_API_KEY",
            "link": "https://openrouter.ai/keys",
        },
    }

    assert auth_method_type(method) == "env_var"
    assert auth_method_env_var_name(method) == "OPENROUTER_API_KEY"
    assert auth_method_link(method) == "https://openrouter.ai/keys"
    assert find_auth_method([method], "ENV_VAR:OPENROUTER_API_KEY") == method


@pytest.mark.asyncio
async def test_authenticate_if_needed_env_var_present_calls_authenticate() -> None:
    conn = AsyncMock()
    methods = [
        {
            "id": "env_var:openrouter_api_key",
            "name": "OpenRouter API key",
            "_meta": {"type": "env_var", "varName": "OPENROUTER_API_KEY"},
        }
    ]

    result = await _authenticate_if_needed(
        conn,
        methods,
        agent_env={"OPENROUTER_API_KEY": "token"},
    )

    assert result.authenticated is True
    assert result.restart is None
    conn.authenticate.assert_awaited_once_with(method_id="env_var:openrouter_api_key")


@pytest.mark.asyncio
async def test_authenticate_if_needed_env_var_missing_returns_restart(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = AsyncMock()
    methods = [
        {
            "id": "env_var:openrouter_api_key",
            "name": "OpenRouter API key",
            "_meta": {
                "type": "env_var",
                "varName": "OPENROUTER_API_KEY",
                "link": "https://openrouter.ai/keys",
            },
        }
    ]
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    result = await _authenticate_if_needed(
        conn,
        methods,
        agent_env={},
        prompt_secret=lambda _prompt: "secret-token",
    )

    assert result.authenticated is False
    assert result.restart is not None
    assert result.restart.method_id == "env_var:openrouter_api_key"
    assert result.restart.env_var_name == "OPENROUTER_API_KEY"
    assert result.restart.env_var_value == "secret-token"
    conn.authenticate.assert_not_awaited()


def test_auth_methods_for_error_prefers_narrowed_list() -> None:
    init_methods = [{"id": "env_var:openrouter_api_key"}, {"id": "openai"}]
    error_data = {"authMethods": [{"id": "openai"}]}

    methods = _auth_methods_for_error(init_methods, error_data)

    assert methods == [{"id": "openai"}]
