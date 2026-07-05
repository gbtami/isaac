from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from acp import RequestError
from acp.agent.connection import AgentSideConnection
from acp import text_block
from acp.schema import AgentMessageChunk

from tests.utils import make_function_agent
from isaac.agent import models as model_registry
from isaac.agent.agent import ACPAgent
from isaac.agent.oauth.openai_codex import model as openai_codex_model
from isaac.agent.slash import handle_slash_command


def _raise_model_error(*_: object, **__: object) -> object:
    raise RuntimeError("boom")


def test_load_runtime_env_reads_shared_config_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    shared_env = tmp_path / "xdg" / "isaac" / ".env"
    shared_env.parent.mkdir(parents=True, exist_ok=True)
    shared_env.write_text("OPENROUTER_API_KEY=shared-key\n", encoding="utf-8")

    monkeypatch.setattr(model_registry, "ENV_FILE", shared_env)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    model_registry.load_runtime_env()

    assert os.getenv("OPENROUTER_API_KEY") == "shared-key"


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
async def test_set_config_option_model_uses_shared_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
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

    shared_env = tmp_path / "xdg" / "isaac" / ".env"
    shared_env.write_text("OPENROUTER_API_KEY=shared-key\n", encoding="utf-8")
    monkeypatch.setattr(model_registry, "ENV_FILE", shared_env)

    conn = AsyncMock(spec=AgentSideConnection)
    conn.session_update = AsyncMock()
    agent = ACPAgent(conn)
    session = await agent.new_session(cwd=str(tmp_path), mcp_servers=[])

    await agent.set_config_option(
        config_id="model",
        session_id=session.session_id,
        value="openrouter:test-openrouter",
    )

    assert agent._session_model_ids[session.session_id] == "openrouter:test-openrouter"


@pytest.mark.asyncio
async def test_set_config_option_model_returns_auth_required_for_missing_provider_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("ALIBABA_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    local_models = tmp_path / "xdg" / "isaac" / "models.json"
    monkeypatch.setattr(model_registry, "LOCAL_MODELS_FILE", local_models)
    local_models.parent.mkdir(parents=True, exist_ok=True)
    local_models.write_text(
        json.dumps(
            {
                "models": {
                    "alibaba:test-qwen": {
                        "provider": "alibaba",
                        "model": "qwen-max",
                        "description": "test alibaba model",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    conn = AsyncMock(spec=AgentSideConnection)
    conn.session_update = AsyncMock()
    agent = ACPAgent(conn)
    session = await agent.new_session(cwd=str(tmp_path), mcp_servers=[])

    with pytest.raises(RequestError) as exc_info:
        await agent.set_config_option(
            config_id="model",
            session_id=session.session_id,
            value="alibaba:test-qwen",
        )

    assert exc_info.value.code == -32000
    assert isinstance(exc_info.value.data, dict)
    methods = exc_info.value.data.get("authMethods")
    assert isinstance(methods, list)
    method_ids = {str(item.get("id")) for item in methods if isinstance(item, dict)}
    assert "env_var:alibaba_api_key" in method_ids
    assert "env_var:dashscope_api_key" in method_ids


@pytest.mark.asyncio
async def test_set_config_option_model_returns_auth_required_for_openai_codex_login(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("HOME", str(tmp_path))

    local_models = tmp_path / "xdg" / "isaac" / "models.json"
    monkeypatch.setattr(model_registry, "LOCAL_MODELS_FILE", local_models)
    local_models.parent.mkdir(parents=True, exist_ok=True)
    local_models.write_text(
        json.dumps(
            {
                "models": {
                    "openai-codex:test": {
                        "provider": "openai-codex",
                        "model": "gpt-5.2-codex",
                        "description": "test openai codex model",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    conn = AsyncMock(spec=AgentSideConnection)
    conn.session_update = AsyncMock()
    agent = ACPAgent(conn)
    session = await agent.new_session(cwd=str(tmp_path), mcp_servers=[])

    with pytest.raises(RequestError) as exc_info:
        await agent.set_config_option(
            config_id="model",
            session_id=session.session_id,
            value="openai-codex:test",
        )

    assert exc_info.value.code == -32000
    assert isinstance(exc_info.value.data, dict)
    methods = exc_info.value.data.get("authMethods")
    assert isinstance(methods, list)
    method_ids = {str(item.get("id")) for item in methods if isinstance(item, dict)}
    assert "openai" in method_ids


@pytest.mark.asyncio
async def test_set_config_option_model_returns_auth_required_for_code_assist_login(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("HOME", str(tmp_path))

    local_models = tmp_path / "xdg" / "isaac" / "models.json"
    monkeypatch.setattr(model_registry, "LOCAL_MODELS_FILE", local_models)
    local_models.parent.mkdir(parents=True, exist_ok=True)
    local_models.write_text(
        json.dumps(
            {
                "models": {
                    "code-assist:test": {
                        "provider": "code-assist",
                        "model": "gemini-2.5-pro",
                        "description": "test code assist model",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    conn = AsyncMock(spec=AgentSideConnection)
    conn.session_update = AsyncMock()
    agent = ACPAgent(conn)
    session = await agent.new_session(cwd=str(tmp_path), mcp_servers=[])

    with pytest.raises(RequestError) as exc_info:
        await agent.set_config_option(
            config_id="model",
            session_id=session.session_id,
            value="code-assist:test",
        )

    assert exc_info.value.code == -32000
    assert isinstance(exc_info.value.data, dict)
    methods = exc_info.value.data.get("authMethods")
    assert isinstance(methods, list)
    method_ids = {str(item.get("id")) for item in methods if isinstance(item, dict)}
    assert "code-assist" in method_ids


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


def test_current_model_falls_back_when_persisted_model_is_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    user_models_file = tmp_path / "xdg" / "isaac" / "models.json"
    settings_file = user_models_file.parent / "isaac.ini"
    user_models_file.parent.mkdir(parents=True, exist_ok=True)
    user_models_file.write_text(json.dumps({"models": {}}), encoding="utf-8")
    settings_file.write_text("[models]\ncurrent_model = openrouter:test-openrouter\n", encoding="utf-8")
    monkeypatch.setattr(model_registry, "USER_MODELS_FILE", user_models_file)

    current = model_registry.current_model_id()

    assert current == model_registry.DEFAULT_MODEL_ID
    assert "current_model = function:function" in settings_file.read_text(encoding="utf-8")


def test_static_catalog_hides_openai_codex_models_by_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    catalog_file = tmp_path / "catalog.json"
    catalog_file.write_text(
        json.dumps(
            {
                "providers": {
                    "openai-codex": {
                        "label": "OpenAI Codex",
                        "models": [{"id": "codex-mini-latest", "name": "Codex Mini"}],
                    },
                    "openai": {
                        "label": "OpenAI",
                        "models": [{"id": "gpt-4.1-mini", "name": "GPT-4.1 mini"}],
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(model_registry, "MODELS_DEV_CATALOG_FILE", catalog_file)
    monkeypatch.delenv(model_registry.OPENAI_CODEX_CATALOG_MODELS_ENV, raising=False)

    models = model_registry.load_models_config()["models"]

    assert "openai:gpt-4.1-mini" in models
    assert "openai-codex:codex-mini-latest" not in models


def test_openai_codex_oauth_models_are_filtered_to_current_chatgpt_codex_list(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    user_models_file = tmp_path / "xdg" / "isaac" / "models.json"
    user_models_file.parent.mkdir(parents=True, exist_ok=True)
    user_models_file.write_text(
        json.dumps(
            {
                "models": {
                    "openai-codex:gpt-5.5": {
                        "provider": "openai-codex",
                        "model": "gpt-5.5",
                        "oauth_source": model_registry.OPENAI_CODEX_OAUTH_SOURCE,
                    },
                    "openai-codex:gpt-5.2-codex": {
                        "provider": "openai-codex",
                        "model": "gpt-5.2-codex",
                        "oauth_source": model_registry.OPENAI_CODEX_OAUTH_SOURCE,
                    },
                    "openai-codex:legacy-catalog-model": {
                        "provider": "openai-codex",
                        "model": "legacy-catalog-model",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(model_registry, "USER_MODELS_FILE", user_models_file)
    monkeypatch.delenv(model_registry.OPENAI_CODEX_CATALOG_MODELS_ENV, raising=False)

    visible = model_registry.list_user_models()

    assert "openai-codex:gpt-5.5" in visible
    assert "openai-codex:gpt-5.2-codex" not in visible
    assert "openai-codex:legacy-catalog-model" not in visible


def test_current_model_falls_back_when_persisted_openai_codex_catalog_model_is_hidden(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    user_models_file = tmp_path / "xdg" / "isaac" / "models.json"
    settings_file = user_models_file.parent / "isaac.ini"
    user_models_file.parent.mkdir(parents=True, exist_ok=True)
    user_models_file.write_text(
        json.dumps(
            {
                "models": {
                    "openai-codex:codex-mini-latest": {
                        "provider": "openai-codex",
                        "model": "codex-mini-latest",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    settings_file.write_text("[models]\ncurrent_model = openai-codex:codex-mini-latest\n", encoding="utf-8")
    monkeypatch.setattr(model_registry, "USER_MODELS_FILE", user_models_file)
    monkeypatch.delenv(model_registry.OPENAI_CODEX_CATALOG_MODELS_ENV, raising=False)

    current = model_registry.current_model_id()

    assert current == model_registry.DEFAULT_MODEL_ID
    assert "current_model = function:function" in settings_file.read_text(encoding="utf-8")


def test_set_current_model_raises_when_persistence_fails(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        model_registry,
        "list_models",
        lambda: {model_registry.DEFAULT_MODEL_ID: {"provider": "function", "model": "function"}},
    )
    monkeypatch.setattr(model_registry, "_save_current_model", lambda _model_id: False)

    with pytest.raises(RuntimeError, match="persist"):
        model_registry.set_current_model(model_registry.DEFAULT_MODEL_ID)


def test_openai_codex_model_sync_prunes_stale_oauth_models(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    models_file = tmp_path / "xdg" / "isaac" / "models.json"
    models_file.parent.mkdir(parents=True, exist_ok=True)
    models_file.write_text(
        json.dumps(
            {
                "models": {
                    "openai-codex:gpt-5.1-codex-mini": {
                        "provider": "openai-codex",
                        "model": "gpt-5.1-codex-mini",
                        "oauth_source": openai_codex_model.OPENAI_CODEX_OAUTH_SOURCE,
                    },
                    "openai-codex:manual": {
                        "provider": "openai-codex",
                        "model": "manual",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(openai_codex_model, "MODELS_FILE", models_file)

    added = openai_codex_model.add_openai_codex_models(["gpt-5.5", "gpt-5.5", "gpt-5.2"])
    stored = json.loads(models_file.read_text(encoding="utf-8"))["models"]

    assert added == 1
    assert "openai-codex:gpt-5.5" in stored
    assert "openai-codex:gpt-5.2" not in stored
    assert "openai-codex:gpt-5.1-codex-mini" not in stored
    assert "openai-codex:manual" in stored


def test_openai_codex_model_filter_normalizes_and_drops_deprecated_models() -> None:
    assert openai_codex_model.normalize_codex_model_name("openai/gpt-5.4") == "gpt-5.4"
    assert openai_codex_model.normalize_codex_model_name("openai-codex:gpt-5.5") == "gpt-5.5"
    assert openai_codex_model.is_supported_chatgpt_codex_model("gpt-5.5")
    assert openai_codex_model.is_supported_chatgpt_codex_model("gpt-5.3-codex-spark")
    assert not openai_codex_model.is_supported_chatgpt_codex_model("gpt-5.2-codex")
    assert not openai_codex_model.is_supported_chatgpt_codex_model("codex-mini-latest")


def test_models_dev_catalog_snapshot_has_no_static_openai_codex_provider() -> None:
    catalog = json.loads(model_registry.MODELS_DEV_CATALOG_FILE.read_text(encoding="utf-8"))

    assert "openai-codex" not in catalog.get("providers", {})


def test_openai_codex_defaults_are_advertised_without_oauth_sync(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    catalog_file = tmp_path / "models_dev_catalog.json"
    catalog_file.write_text(json.dumps({"providers": {}}), encoding="utf-8")
    monkeypatch.setattr(model_registry, "MODELS_DEV_CATALOG_FILE", catalog_file)
    monkeypatch.setattr(model_registry, "MODELS_DEV_SNAPSHOT_FILE", tmp_path / "models_dev_api.json")
    monkeypatch.setattr(model_registry, "LOCAL_MODELS_FILE", tmp_path / "missing-local-models.json")
    monkeypatch.setattr(model_registry, "USER_MODELS_FILE", tmp_path / "missing-user-models.json")

    models = model_registry.list_user_models()

    assert "openai-codex:gpt-5.5" in models
    assert "openai-codex:gpt-5.4" in models
    assert "openai-codex:gpt-5.4-mini" in models
    assert "openai-codex:gpt-5.2-codex" not in models


def test_openai_codex_defaults_repair_existing_local_entry_for_selector(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    catalog_file = tmp_path / "models_dev_catalog.json"
    catalog_file.write_text(json.dumps({"providers": {}}), encoding="utf-8")
    user_models_file = tmp_path / "models.json"
    user_models_file.write_text(
        json.dumps(
            {
                "models": {
                    "openai-codex:gpt-5.4-mini": {
                        "provider": "openai-codex",
                        "model": "gpt-5.4-mini",
                        "description": "custom description",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(model_registry, "MODELS_DEV_CATALOG_FILE", catalog_file)
    monkeypatch.setattr(model_registry, "MODELS_DEV_SNAPSHOT_FILE", tmp_path / "models_dev_api.json")
    monkeypatch.setattr(model_registry, "LOCAL_MODELS_FILE", tmp_path / "missing-local-models.json")
    monkeypatch.setattr(model_registry, "USER_MODELS_FILE", user_models_file)

    models = model_registry.list_user_models()

    entry = models["openai-codex:gpt-5.4-mini"]
    assert entry["oauth_source"] == model_registry.OPENAI_CODEX_OAUTH_SOURCE
    assert entry["description"] == "custom description"


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
