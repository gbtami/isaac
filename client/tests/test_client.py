"""Exercise client startup, choices, and session replay over real ACP stdio."""

import asyncio
import os
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from isaac_client import cli
from isaac_client.display import Terminal
from isaac_client.history import SessionHistory


@pytest.fixture
def command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    monkeypatch.setenv("ISAAC_CLIENT_STATE_DIR", str(tmp_path / "client-state"))
    return [sys.executable, str(Path(__file__).with_name("agent_fixture.py"))]


class ScriptedTerminal(Terminal):
    def __init__(self, prompts: list[str], selections: list[int]) -> None:
        super().__init__()
        self.interactive = True
        self.prompts = iter(prompts)
        self.selections = iter(selections)
        self.menus: dict[str, list[str]] = {}
        self.toolbars: list[str] = []

    async def read(
        self, label: str, *, commands: list[str] | None = None, toolbar: Callable[[], str] | None = None
    ) -> str:
        if toolbar:
            self.toolbars.append(toolbar())
        return next(self.prompts, "/client:quit")

    async def choose(self, label: str, choices: list[str]) -> int | None:
        self.menus[label] = choices
        return next(self.selections, None)


async def test_config_approval_and_named_resume(
    command: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    terminal = ScriptedTerminal(
        ["/client:help", "/client:model", "/client:config", "Demo prompt", "/client:plan"],
        [1, 1, 0, 1],
    )
    monkeypatch.setattr(cli, "Terminal", lambda: terminal)

    async with asyncio.timeout(15):
        assert await cli.run_client(command, tmp_path, []) == 0
        output = capsys.readouterr().out
        assert "/demo  Run the demo" in output
        assert "model=two verbose=True choice=reject" in output
        assert "Check the result" in output
        assert "Approve tool call: Edit demo.txt" in output
        assert "before" in output and "after" in output
        assert terminal.menus["Available models"] == ["Model One (current)", "Model Two"]
        assert any("Context 25/1,000" in toolbar for toolbar in terminal.toolbars)

        history = SessionHistory(command, tmp_path)
        assert [(entry.session_id, entry.title) for entry in history.recent()] == [("demo", "Agent-assigned title")]
        assert SessionHistory([*command, "--another-agent-profile"], tmp_path).recent() == []
        terminal = ScriptedTerminal([], [0])
        assert await cli.run_client(command, tmp_path, [], resume=True) == 0
        assert terminal.menus["Resume session"] == ["Agent-assigned title  [demo]"]
        assert "Session history replayed over ACP." in capsys.readouterr().out


async def test_cli_cancels_permission_without_terminal(command: list[str], tmp_path: Path) -> None:
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "isaac_client",
        "--project-dir",
        str(tmp_path),
        *command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=os.environ,
    )
    async with asyncio.timeout(15):
        stdout, stderr = await process.communicate(b"Test the approval\n")
    assert process.returncode == 0, stderr.decode()
    assert b"choice=cancelled" in stdout
