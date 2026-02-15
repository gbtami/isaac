from __future__ import annotations

import json
import os
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


def _raise_model_error(*_: object, **__: object) -> object:
    raise RuntimeError("boom")


def test_load_runtime_env_prefers_session_cwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    shared_env = tmp_path / "xdg" / "isaac" / ".env"
    shared_env.parent.mkdir(parents=True, exist_ok=True)
    shared_env.write_text("OPENROUTER_API_KEY=shared-key\n", encoding="utf-8")
    project_dir = tmp_path / "project"
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / ".env").write_text("OPENROUTER_API_KEY=session-key\n", encoding="utf-8")

    monkeypatch.setattr(model_registry, "ENV_FILE", shared_env)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    model_registry.load_runtime_env(project_dir)

    assert os.getenv("OPENROUTER_API_KEY") == "session-key"


@pytest.mark.asyncio
async def test_set_config_option_model_changes_runner(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(model_registry, "LOCAL_MODELS_FILE", tmp_path / "xdg" / "isaac" / "models.json")
    model_registry.LOCAL_MODELS_FILE.parent.mkdir(parents=True, exist_ok=True)
    conn = AsyncMock(spec=AgentSideConnection)
    agent = make_function_agent(conn)
    session = await agent.new_session(cwd="/", mcp_servers=[])

    await agent.set_config_option(
        config_id="model",
        session_id=session.session_id,
        value="function:function",
    )

    response = await agent.prompt(prompt=[text_block("hello")], session_id=session.session_id)

    conn.session_update.assert_called()
    updates = [call.kwargs["update"] for call in conn.session_update.call_args_list]  # type: ignore[attr-defined]
    agent_chunks = [u for u in updates if isinstance(u, AgentMessageChunk)]
    assert agent_chunks
    assert any(getattr(c.content, "text", "") and "Error" not in getattr(c.content, "text", "") for c in agent_chunks)
    assert response.stop_reason == "end_turn"


@pytest.mark.asyncio
async def test_set_config_option_model_uses_session_cwd_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    local_models = tmp_path / "xdg" / "isaac" / "models.json"
    monkeypatch.setattr(model_registry, "LOCAL_MODELS_FILE", local_models)
    local_models.parent.mkdir(parents=True, exist_ok=True)
    local_models.write_text(
        json.dumps(
            {
                "models": {
                    "openrouter:test-openrouter": {
                        "provider": "openrouter",
                        "model": "openai/gpt-5",
                        "description": "test openrouter model",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    project_dir = tmp_path / "project"
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / ".env").write_text("OPENROUTER_API_KEY=session-key\n", encoding="utf-8")

    conn = AsyncMock(spec=AgentSideConnection)
    conn.session_update = AsyncMock()
    agent = ACPAgent(conn)
    session = await agent.new_session(cwd=str(project_dir), mcp_servers=[])

    await agent.set_config_option(
        config_id="model",
        session_id=session.session_id,
        value="openrouter:test-openrouter",
    )

    assert agent._session_model_ids[session.session_id] == "openrouter:test-openrouter"


@pytest.mark.asyncio
async def test_model_build_failure_surfaces_error(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setattr("isaac.agent.brain.session_ops.create_subagent_for_model", _raise_model_error)

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


def test_google_uses_gemini_api_key_fallback(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    _model, settings = model_registry._build_provider_model(  # type: ignore[attr-defined]
        "google:gemini-2.5-pro",
        {"provider": "google", "model": "gemini-2.5-pro"},
    )
    assert isinstance(settings, dict)
    assert settings.get("google_thinking_config") == {"include_thoughts": True}


def test_azure_requires_endpoint(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    with pytest.raises(RuntimeError, match="AZURE_OPENAI_ENDPOINT"):
        model_registry._build_provider_model(  # type: ignore[attr-defined]
            "azure:gpt-4.1",
            {"provider": "azure", "model": "gpt-4.1"},
        )


@pytest.mark.parametrize(
    ("provider", "required_env"),
    [
        ("alibaba", "ALIBABA_API_KEY"),
        ("cohere", "CO_API_KEY"),
        ("deepseek", "DEEPSEEK_API_KEY"),
        ("fireworks", "FIREWORKS_API_KEY"),
        ("github", "GITHUB_API_KEY"),
        ("groq", "GROQ_API_KEY"),
        ("huggingface", "HF_TOKEN"),
        ("moonshotai", "MOONSHOTAI_API_KEY"),
        ("nebius", "NEBIUS_API_KEY"),
        ("ovhcloud", "OVHCLOUD_API_KEY"),
        ("together", "TOGETHER_API_KEY"),
        ("xai", "XAI_API_KEY"),
    ],
)
def test_new_provider_branches_require_expected_env(monkeypatch: pytest.MonkeyPatch, provider: str, required_env: str):
    monkeypatch.delenv(required_env, raising=False)
    with pytest.raises(RuntimeError, match=required_env):
        model_registry._build_provider_model(  # type: ignore[attr-defined]
            f"{provider}:test-model",
            {"provider": provider, "model": "test-model"},
        )
