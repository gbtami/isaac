from __future__ import annotations

import os

from isaac.agent import models as model_registry


def test_models_paths_use_test_xdg_home() -> None:
    config_home = os.environ.get("XDG_CONFIG_HOME", "")
    assert config_home, "XDG_CONFIG_HOME must be set in tests"
    assert str(model_registry.ENV_FILE).startswith(config_home)
    assert str(model_registry.USER_MODELS_FILE).startswith(config_home)
