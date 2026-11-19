"""Exercise client startup, choices, and session replay over real ACP stdio."""

import asyncio
import os
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock

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
        self,
        label: str,
        *,
        commands: list[str] | None = None,
        toolbar: Callable[[], str] | None = None,
        default: str = "",
    ) -> str:
        if label.startswith("Queue"):
            return await asyncio.Future[str]()
        if toolbar:
            self.toolbars.append(toolbar())
        return next(self.prompts, "/client:quit")

    async def choose(
        self, label: str, choices: list[str], *, toolbar: Callable[[], str] | None = None, search: bool = False
    ) -> int | None:
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
        assert terminal.menus["Resume session"][0].startswith("Agent-assigned title  [demo]")
        assert "Session history replayed over ACP." in capsys.readouterr().out


async def test_one_stdout_proxy_across_interactive_turns(
    command: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    terminal = ScriptedTerminal(["Echo first", "Echo second"], [])
    monkeypatch.setattr(cli, "Terminal", lambda: terminal)
    proxy_events: list[str] = []

    @contextmanager
    def track_stdout_proxy() -> Iterator[None]:
        proxy_events.append("opened")
        try:
            yield
        finally:
            proxy_events.append("closed")

    monkeypatch.setattr(cli, "patch_stdout", track_stdout_proxy)
    async with asyncio.timeout(15):
        assert await cli.run_client(command, tmp_path, []) == 0
    assert proxy_events == ["opened", "closed"]
    output = capsys.readouterr().out
    assert "SDK Demo: Echo first" in output
    assert "SDK Demo: Echo second" in output


async def test_resume_filters_choices_by_title(monkeypatch: pytest.MonkeyPatch) -> None:
    terminal = Terminal()
    terminal.interactive = True
    monkeypatch.setattr(terminal, "read", AsyncMock(side_effect=["BUILD", "1"]))
    assert await terminal.choose("Resume session", ["Write docs", "Fix the build"], search=True) == 1


async def test_back_to_back_menus_take_priority_over_the_editor(monkeypatch: pytest.MonkeyPatch) -> None:
    terminal = Terminal()
    terminal.interactive = True
    editor_started = asyncio.Event()

    async def read(*args: object, **kwargs: object) -> str:
        editor_started.set()
        return await asyncio.Future[str]()

    async def another_menu() -> None:
        async with terminal.menu():
            assert not editor_started.is_set()

    monkeypatch.setattr(terminal, "read", read)
    async with asyncio.timeout(5):
        async with terminal.menu():
            editor = asyncio.create_task(terminal.edit("Queue ❯ ", commands=[], toolbar=lambda: ""))
            await asyncio.sleep(0)
            menu = asyncio.create_task(another_menu())
            await asyncio.sleep(0)
        try:
            await menu
            await editor_started.wait()
        finally:
            editor.cancel()
            await asyncio.gather(editor, return_exceptions=True)


async def test_continue_discovers_paginated_agent_history(
    command: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "Terminal", lambda: ScriptedTerminal([], []))
    async with asyncio.timeout(15):
        assert await cli.run_client([*command, "--list-sessions"], tmp_path, [], continue_latest=True) == 0
    output = capsys.readouterr().out
    assert "Session remote-new" in output
    assert "Session history replayed over ACP." in output


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


async def test_groups_reads_and_keeps_failures_visible(
    command: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "Terminal", lambda: ScriptedTerminal(["Show tool activity"], []))
    async with asyncio.timeout(15):
        assert await cli.run_client(command, tmp_path, []) == 0
    output = capsys.readouterr().out
    assert output.count("Completed: read 2") == 1
    assert "Routine output" not in output
    assert "Failed: Run tests" in output
    assert "Test failed with details" in output


async def test_shows_prompt_error_details_and_accepts_another_turn(
    command: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "Terminal", lambda: ScriptedTerminal(["Fail the turn", "Echo recovered"], []))
    async with asyncio.timeout(15):
        assert await cli.run_client(command, tmp_path, []) == 0
    output = capsys.readouterr().out
    assert "Internal error: Set the provider credentials and try again." in output
    assert "Echo recovered" in output


async def test_reviews_and_edits_structured_answers(
    command: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    terminal = ScriptedTerminal(["Ask a form question", "1", "", "e", "2", "Use the LTS release", "y"], [])
    monkeypatch.setattr(cli, "Terminal", lambda: terminal)
    async with asyncio.timeout(15):
        assert await cli.run_client(command, tmp_path, []) == 0
    output = capsys.readouterr().out
    assert "SDK Demo asks: Which runtime should the example use?" in output
    assert output.count("Review answers") == 2
    assert "Answer: {'runtime': 'JavaScript', 'note': 'Use the LTS release'}" in output


@pytest.mark.parametrize("cancel", [False, True])
async def test_can_queue_or_cancel_a_busy_turn(
    command: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    cancel: bool,
) -> None:
    started = asyncio.Event()
    seen_input = False

    class BusyTerminal(ScriptedTerminal):
        def write(self, text: str, *, end: str = "\n", style: str | None = None) -> None:
            super().write(text, end=end, style=style)
            if text == "Turn started":
                started.set()

        async def read(
            self,
            label: str,
            *,
            commands: list[str] | None = None,
            toolbar: Callable[[], str] | None = None,
            default: str = "",
        ) -> str:
            nonlocal seen_input
            if label.startswith("Queue") and not seen_input:
                await started.wait()
                seen_input = True
                if cancel:
                    raise KeyboardInterrupt
                return "Echo queued follow-up"
            return await super().read(label, commands=commands, toolbar=toolbar, default=default)

    prompts = ["Wait for cancel", "Echo after cancellation"] if cancel else ["Wait then end"]
    terminal = BusyTerminal(prompts, [])
    monkeypatch.setattr(cli, "Terminal", lambda: terminal)
    async with asyncio.timeout(15):
        assert await cli.run_client(command, tmp_path, []) == 0
    output = capsys.readouterr().out
    if cancel:
        assert "Stopped: cancelled" in output
        assert "Echo after cancellation" in output
    else:
        assert "Queued follow-up 1." in output
        assert output.index("Turn finished") < output.index("Echo queued follow-up")
