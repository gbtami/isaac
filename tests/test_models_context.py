from __future__ import annotations

import json
from pathlib import Path

import pytest

from isaac.agent import models as model_registry


@pytest.mark.asyncio
async def test_context_limit_applied_from_offline_snapshot(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("HOME", str(tmp_path))

    local_models_file = tmp_path / "models.json"
    local_models_file.write_text(
        json.dumps(
            {
                "models": {
                    "function:function": {
                        "provider": "function",
                        "model": "function",
                        "description": "In-process function model for deterministic testing",
                    },
                    "openai:gpt-4o-mini": {
                        "provider": "openai",
                        "model": "gpt-4o-mini",
                        "description": "OpenAI GPT-4o mini",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(model_registry, "LOCAL_MODELS_FILE", local_models_file)
    monkeypatch.setattr(model_registry, "USER_MODELS_FILE", tmp_path / "xdg" / "isaac" / "models.json")

    snapshot_file = tmp_path / "models_dev_api.json"
    snapshot_file.write_text(
        json.dumps(
            {
                "openai": {
                    "models": {
                        "gpt-4o-mini": {"limit": {"context": 128000}},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(model_registry, "MODELS_DEV_SNAPSHOT_FILE", snapshot_file)

    config = model_registry.load_models_config()
    entry = config["models"]["openai:gpt-4o-mini"]
    assert entry.get("context_limit") == 128000


def test_offline_catalog_contains_newly_supported_providers():
    catalog = json.loads(model_registry.MODELS_DEV_CATALOG_FILE.read_text(encoding="utf-8"))
    providers = catalog.get("providers", {})
    for provider in (
        "alibaba",
        "azure",
        "bedrock",
        "cohere",
        "deepseek",
        "fireworks",
        "github",
        "google-vertex",
        "groq",
        "huggingface",
        "moonshotai",
        "nebius",
        "ovhcloud",
        "together",
        "vercel",
        "xai",
    ):
        assert provider in providers
        models = providers[provider].get("models", [])
        assert isinstance(models, list)
        assert models
