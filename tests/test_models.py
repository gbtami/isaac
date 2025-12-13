from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from acp.agent.connection import AgentSideConnection
from acp import text_block
from acp.schema import AgentMessageChunk

from tests.utils import make_function_agent
from isaac.agent import models as model_registry
from isaac.agent.agent import ACPAgent
from isaac.agent.slash import handle_slash_command


def _raise_model_error(*_: object, **__: object) -> tuple[object, object]:
    raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_set_session_model_changes_runner(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(model_registry, "MODELS_FILE", tmp_path / "xdg" / "isaac" / "models.json")
    model_registry.MODELS_FILE.parent.mkdir(parents=True, exist_ok=True)
    conn = AsyncMock(spec=AgentSideConnection)
    agent = make_function_agent(conn)
    session = await agent.new_session(cwd="/", mcp_servers=[])

    await agent.set_session_model(model_id="function:function", session_id=session.session_id)

    response = await agent.prompt(prompt=[text_block("hello")], session_id=session.session_id)

    conn.session_update.assert_called()
    updates = [call.kwargs["update"] for call in conn.session_update.call_args_list]  # type: ignore[attr-defined]
    agent_chunks = [u for u in updates if isinstance(u, AgentMessageChunk)]
    assert agent_chunks
    assert any(getattr(c.content, "text", "") and "Error" not in getattr(c.content, "text", "") for c in agent_chunks)
    assert response.stop_reason == "end_turn"


@pytest.mark.asyncio
async def test_model_build_failure_surfaces_error(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setattr(model_registry, "build_agent_pair", _raise_model_error)

    conn = AsyncMock(spec=AgentSideConnection)
    conn.session_update = AsyncMock()
    agent = ACPAgent(conn)
    session = await agent.new_session(cwd=str(tmp_path), mcp_servers=[])

    updates = [call.kwargs["update"] for call in conn.session_update.await_args_list]  # type: ignore[attr-defined]
    assert any(
        isinstance(u, AgentMessageChunk) and "Model load failed" in getattr(getattr(u, "content", None), "text", "")
        for u in updates
    )

    response = await agent.prompt(prompt=[text_block("hi")], session_id=session.session_id)
    assert response.stop_reason == "refusal"


@pytest.mark.asyncio
async def test_unknown_model_id_is_rejected(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    conn = AsyncMock(spec=AgentSideConnection)
    conn.session_update = AsyncMock()
    agent = make_function_agent(conn)
    session = await agent.new_session(cwd=str(tmp_path), mcp_servers=[])

    slash = await handle_slash_command(agent, session.session_id, "/model no-such")
    assert slash is not None
    await agent._send_update(slash)  # type: ignore[attr-defined]

    updates = [call.kwargs["update"] for call in conn.session_update.await_args_list]  # type: ignore[attr-defined]
    assert any(
        isinstance(u, AgentMessageChunk)
        and "Unknown model id: no-such" in getattr(getattr(u, "content", None), "text", "")
        for u in updates
    )


def test_openai_reasoning_effort_enabled_for_reasoning_models(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    _model, settings = model_registry._build_provider_model(  # type: ignore[attr-defined]
        "openai:o1-mini",
        {"provider": "openai", "model": "o1-mini"},
    )
    assert isinstance(settings, dict)
    assert settings.get("openai_reasoning_effort") == "medium"


def test_openai_reasoning_effort_not_set_for_non_reasoning_models(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    _model, settings = model_registry._build_provider_model(  # type: ignore[attr-defined]
        "openai:gpt-4o-mini",
        {"provider": "openai", "model": "gpt-4o-mini"},
    )
    assert settings is None


def test_anthropic_thinking_enabled_by_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    _model, settings = model_registry._build_provider_model(  # type: ignore[attr-defined]
        "anthropic:claude-sonnet-4-0",
        {"provider": "anthropic", "model": "claude-sonnet-4-0"},
    )
    assert isinstance(settings, dict)
    assert settings.get("anthropic_thinking") == {"type": "enabled", "budget_tokens": 512}


def test_google_thinking_enabled_by_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test")
    _model, settings = model_registry._build_provider_model(  # type: ignore[attr-defined]
        "google:gemini-2.5-pro",
        {"provider": "google", "model": "gemini-2.5-pro"},
    )
    assert isinstance(settings, dict)
    assert settings.get("google_thinking_config") == {"include_thoughts": True}


def test_openrouter_reasoning_enabled_by_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    _model, settings = model_registry._build_provider_model(  # type: ignore[attr-defined]
        "openrouter:openai/gpt-5",
        {"provider": "openrouter", "model": "openai/gpt-5"},
    )
    assert isinstance(settings, dict)
    assert settings.get("openrouter_reasoning") == {"effort": "medium"}
