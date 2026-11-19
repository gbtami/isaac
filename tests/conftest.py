"""Keep automatic session persistence out of the developer's state directory."""

from asyncio.subprocess import Process
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Any

import acp
import pytest
from acp.client.connection import ClientSideConnection


@pytest.fixture(autouse=True)
def isolate_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state_dir = str(tmp_path / "state")
    monkeypatch.setenv("ISAAC_STATE_DIR", state_dir)
    spawn = acp.spawn_agent_process

    def spawn_with_state(
        *args: Any, **kwargs: Any
    ) -> AbstractAsyncContextManager[tuple[ClientSideConnection, Process]]:
        # The SDK only inherits an allowlist of variables by default.
        kwargs["env"] = {"ISAAC_STATE_DIR": state_dir, **(kwargs.get("env") or {})}
        return spawn(*args, **kwargs)

    monkeypatch.setattr(acp, "spawn_agent_process", spawn_with_state)
