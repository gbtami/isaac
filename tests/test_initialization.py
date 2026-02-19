from __future__ import annotations

from types import SimpleNamespace
import pytest
from acp import PROTOCOL_VERSION, text_block
from acp import RequestError
from acp.agent.connection import AgentSideConnection
from acp.schema import AgentMessageChunk
from unittest.mock import AsyncMock

from tests.utils import make_function_agent, make_error_agent, make_timeout_agent


@pytest.mark.asyncio
async def test_initialize_includes_tools():
    conn = AsyncMock(spec=AgentSideConnection)
    agent = make_function_agent(conn)

    response = await agent.initialize(protocol_version=PROTOCOL_VERSION)

    assert response.protocol_version == PROTOCOL_VERSION
    assert response.agent_info.name == "isaac"


@pytest.mark.asyncio
async def test_prompt_echoes_plain_text():
    conn = AsyncMock(spec=AgentSideConnection)
    agent = make_function_agent(conn)

    session_id = "test-session"
    response = await agent.prompt(prompt=[text_block("hello world")], session_id=session_id)

    assert conn.session_update.call_count >= 1
    messages = [
        call.kwargs["update"]
        for call in conn.session_update.call_args_list
        if isinstance(call.kwargs.get("update"), AgentMessageChunk)
    ]
    assert messages, "Expected at least one AgentMessageChunk"
    assert any(getattr(m.content, "text", "") for m in messages)
    assert response.stop_reason == "end_turn"


@pytest.mark.asyncio
async def test_provider_error_is_sent_to_client():
    conn = AsyncMock(spec=AgentSideConnection)
    agent = make_error_agent(conn)

    session_id = "err-session"
    response = await agent.prompt(prompt=[text_block("do something")], session_id=session_id)

    assert response.stop_reason == "end_turn"
    updates = [
        call.kwargs["update"]
        for call in conn.session_update.call_args_list
        if isinstance(call.kwargs.get("update"), AgentMessageChunk)
    ]
    assert any("Model/provider error" in getattr(u.content, "text", "") for u in updates)


@pytest.mark.asyncio
async def test_provider_timeout_is_sent_to_client():
    conn = AsyncMock(spec=AgentSideConnection)
    agent = make_timeout_agent(conn)

    session_id = "timeout-session"
    response = await agent.prompt(prompt=[text_block("do something")], session_id=session_id)

    assert response.stop_reason == "end_turn"
    updates = [
        call.kwargs["update"]
        for call in conn.session_update.call_args_list
        if isinstance(call.kwargs.get("update"), AgentMessageChunk)
    ]
    assert any("Model/provider timeout" in getattr(u.content, "text", "") for u in updates)


@pytest.mark.asyncio
async def test_initialize_negotiates_protocol_version():
    conn = AsyncMock(spec=AgentSideConnection)
    agent = make_function_agent(conn)

    resp = await agent.initialize(protocol_version=PROTOCOL_VERSION + 1)
    assert resp.protocol_version == PROTOCOL_VERSION


@pytest.mark.asyncio
async def test_initialize_advertises_auth_methods():
    conn = AsyncMock(spec=AgentSideConnection)
    agent = make_function_agent(conn)

    resp = await agent.initialize(protocol_version=PROTOCOL_VERSION)

    assert resp.auth_methods is not None
    method_ids = {item.id for item in resp.auth_methods}
    assert {"openai", "code-assist"}.issubset(method_ids)
    env_methods = [
        method.model_dump(by_alias=True, exclude_none=True)
        for method in resp.auth_methods
        if method.id.startswith("env_var:")
    ]
    assert env_methods
    assert any((item.get("_meta") or {}).get("type") == "env_var" for item in env_methods)


@pytest.mark.asyncio
async def test_authenticate_rejects_unknown_method():
    conn = AsyncMock(spec=AgentSideConnection)
    agent = make_function_agent(conn)

    with pytest.raises(RequestError) as exc_info:
        await agent.authenticate(method_id="unknown")

    assert exc_info.value.code == -32602


@pytest.mark.asyncio
async def test_authenticate_custom_method_succeeds():
    conn = AsyncMock(spec=AgentSideConnection)
    agent = make_function_agent(conn, auth_methods=[{"id": "custom", "name": "Custom"}])

    resp = await agent.authenticate(method_id="custom")

    assert resp is not None


@pytest.mark.asyncio
async def test_authenticate_env_var_method_requires_variable(monkeypatch):
    conn = AsyncMock(spec=AgentSideConnection)
    agent = make_function_agent(
        conn,
        auth_methods=[
            {
                "id": "env_var:test",
                "name": "Test key",
                "_meta": {"type": "env_var", "varName": "ISAAC_TEST_REQUIRED_KEY"},
            }
        ],
    )
    monkeypatch.delenv("ISAAC_TEST_REQUIRED_KEY", raising=False)

    with pytest.raises(RequestError) as exc_info:
        await agent.authenticate(method_id="env_var:test")

    assert exc_info.value.code == -32000
    assert isinstance(exc_info.value.data, dict)
    assert exc_info.value.data.get("missingEnvVar") == "ISAAC_TEST_REQUIRED_KEY"


@pytest.mark.asyncio
async def test_authenticate_env_var_method_succeeds_when_variable_present(monkeypatch):
    conn = AsyncMock(spec=AgentSideConnection)
    agent = make_function_agent(
        conn,
        auth_methods=[
            {
                "id": "env_var:test",
                "name": "Test key",
                "_meta": {"type": "env_var", "varName": "ISAAC_TEST_REQUIRED_KEY"},
            }
        ],
    )
    monkeypatch.setenv("ISAAC_TEST_REQUIRED_KEY", "token")

    resp = await agent.authenticate(method_id="env_var:test")

    assert resp is not None


@pytest.mark.asyncio
async def test_authenticate_openai_oauth_path(monkeypatch):
    conn = AsyncMock(spec=AgentSideConnection)
    agent = make_function_agent(conn)
    calls: dict[str, object] = {}

    class _OpenAISession:
        auth_url = "https://example.com/openai"

        async def exchange_tokens(self):
            calls["exchange"] = True
            return {"access_token": "tok"}

    async def _begin():
        calls["begin"] = True
        return _OpenAISession()

    async def _sync(tokens):
        calls["sync"] = tokens
        return SimpleNamespace(added=0, total=0, used_fallback=False, error=None)

    monkeypatch.setattr("isaac.agent.acp.auth_flow.has_openai_tokens", lambda: False)
    monkeypatch.setattr("isaac.agent.acp.auth_flow.begin_openai_oauth", _begin)
    monkeypatch.setattr("isaac.agent.acp.auth_flow.maybe_open_openai", lambda _url: True)
    monkeypatch.setattr("isaac.agent.acp.auth_flow.save_openai_tokens", lambda tokens: calls.setdefault("save", tokens))
    monkeypatch.setattr("isaac.agent.acp.auth_flow.sync_openai_codex_models", _sync)

    resp = await agent.authenticate(method_id="openai")

    assert resp is not None
    assert calls.get("begin") is True
    assert calls.get("exchange") is True
    assert "save" in calls
    assert "sync" in calls


@pytest.mark.asyncio
async def test_authenticate_code_assist_oauth_path(monkeypatch):
    conn = AsyncMock(spec=AgentSideConnection)
    agent = make_function_agent(conn)
    calls: dict[str, object] = {}

    class _CodeAssistSession:
        auth_url = "https://example.com/code-assist"

        async def exchange_tokens(self):
            calls["exchange"] = True
            return {"access_token": "tok"}

    async def _begin():
        calls["begin"] = True
        return _CodeAssistSession()

    async def _finalize(tokens):
        calls["finalize"] = tokens
        return tokens

    monkeypatch.setattr("isaac.agent.acp.auth_flow.has_code_assist_tokens", lambda: False)
    monkeypatch.setattr("isaac.agent.acp.auth_flow.begin_code_assist_oauth", _begin)
    monkeypatch.setattr("isaac.agent.acp.auth_flow.maybe_open_code_assist", lambda _url: True)
    monkeypatch.setattr("isaac.agent.acp.auth_flow.finalize_code_assist_login", _finalize)

    resp = await agent.authenticate(method_id="code-assist")

    assert resp is not None
    assert calls.get("begin") is True
    assert calls.get("exchange") is True
    assert "finalize" in calls


@pytest.mark.asyncio
async def test_new_session_works_without_authenticate(monkeypatch, tmp_path):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    conn = AsyncMock(spec=AgentSideConnection)
    agent = make_function_agent(conn)
    session = await agent.new_session(cwd=str(tmp_path), mcp_servers=[])

    assert session.session_id


@pytest.mark.asyncio
async def test_new_session_requires_absolute_cwd(monkeypatch, tmp_path):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    conn = AsyncMock(spec=AgentSideConnection)
    agent = make_function_agent(conn)

    with pytest.raises(RequestError):
        await agent.new_session(cwd="relative/path", mcp_servers=[])


@pytest.mark.asyncio
async def test_load_session_requires_absolute_cwd(monkeypatch, tmp_path):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    conn = AsyncMock(spec=AgentSideConnection)
    agent = make_function_agent(conn)

    session_id = "sess-123"
    with pytest.raises(RequestError):
        await agent.load_session(cwd="relative/path", mcp_servers=[], session_id=session_id)
