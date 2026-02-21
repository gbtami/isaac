from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

_TEST_HOME = Path(tempfile.mkdtemp(prefix="isaac-test-home-"))


def _set_isolated_env(base: Path) -> None:
    os.environ["HOME"] = str(base)
    os.environ["XDG_CONFIG_HOME"] = str(base / ".config")
    os.environ["XDG_STATE_HOME"] = str(base / ".local" / "state")
    os.environ["XDG_DATA_HOME"] = str(base / ".local" / "share")
    os.environ["XDG_CACHE_HOME"] = str(base / ".cache")


# Set HOME/XDG eagerly at import time so app modules that compute config/state
# paths during import cannot resolve to the developer's real directories.
_set_isolated_env(_TEST_HOME)


@pytest.fixture(autouse=True)
def _isolate_home(monkeypatch):
    """Force tests to use a temporary HOME/XDG dirs to avoid permission issues."""
    _set_isolated_env(_TEST_HOME)
    monkeypatch.setattr(Path, "home", lambda: _TEST_HOME)
