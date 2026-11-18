"""Keep automatic session persistence out of the developer's state directory."""

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolate_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ISAAC_STATE_DIR", str(tmp_path / "state"))
