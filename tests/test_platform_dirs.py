from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock

from isaac import paths
from isaac.agent.agent import ACPAgent


def test_platform_dirs_use_xdg_homes() -> None:
    expected_config = Path(os.environ["XDG_CONFIG_HOME"]) / "isaac"
    expected_state = Path(os.environ["XDG_STATE_HOME"]) / "isaac"
    expected_cache = Path(os.environ["XDG_CACHE_HOME"]) / "isaac"

    assert paths.config_dir() == expected_config
    assert paths.state_dir() == expected_state
    assert paths.cache_dir() == expected_cache
    assert paths.log_dir().is_relative_to(expected_state)


def test_agent_session_store_uses_state_dir() -> None:
    agent = ACPAgent(conn=AsyncMock())
    assert agent._session_store.root == paths.state_dir() / "sessions"  # type: ignore[attr-defined]
