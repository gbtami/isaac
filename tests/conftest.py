from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path_factory, monkeypatch):
    """Force tests to use a temporary HOME/XDG dirs to avoid permission issues."""
    base = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(base))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(base / ".config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(base / ".local" / "state"))
    monkeypatch.setenv("XDG_DATA_HOME", str(base / ".local" / "share"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(base / ".cache"))
    monkeypatch.setattr(Path, "home", lambda: base)
