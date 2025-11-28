from __future__ import annotations

from unittest.mock import patch

import pytest

from isaac.cli import _select_model_cli
from isaac import models as model_registry


@pytest.mark.asyncio
async def test_select_model_cli_uses_async_dialog_when_loop_running(monkeypatch):
    models = {
        "test": {"description": "Test model"},
        "function-model": {"description": "Function model"},
    }
    current = "test"

    async def fake_run_async():
        return "function-model"

    class FakeDialog:
        def run(self):
            raise RuntimeError("asyncio.run() cannot be called from a running event loop")

    async def run_async(self):
        return await fake_run_async()

    with patch("prompt_toolkit.shortcuts.radiolist_dialog", return_value=FakeDialog()):
        selection = await _select_model_cli(models, current, selection_fallback="function-model")
        assert selection == "function-model"


@pytest.mark.asyncio
async def test_build_agent_function_model(monkeypatch):
    config = model_registry.load_models_config()
    assert "function-model" in config["models"]
    # ensure no exception when building function model
    agent_instance = model_registry.build_agent("function-model", lambda a: None)
    assert agent_instance is not None
