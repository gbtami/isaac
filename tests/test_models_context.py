from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from isaac.agent import models as model_registry


@pytest.mark.asyncio
async def test_context_limit_applied_on_first_write(monkeypatch, tmp_path: Path):
    # Redirect config paths
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(model_registry, "MODELS_FILE", tmp_path / "models.json")

    sample = {
        "openai": {
            "models": {
                "gpt-4o-mini": {"limit": {"context": 128000}},
            }
        }
    }

    def fake_urlopen(url, timeout=5):
        return io.BytesIO(json.dumps(sample).encode("utf-8"))

    monkeypatch.setattr(model_registry.urllib.request, "urlopen", fake_urlopen)  # type: ignore[attr-defined]

    config = model_registry.load_models_config()
    entry = config["models"]["openai:gpt-4o-mini"]
    assert entry.get("context_limit") == 128000
    # Ensure file was written with the new field
    on_disk = json.loads(model_registry.MODELS_FILE.read_text())
    assert on_disk["models"]["openai:gpt-4o-mini"]["context_limit"] == 128000
