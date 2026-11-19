"""Exercise client startup, choices, and session replay over real ACP stdio."""

import asyncio
import io
import os
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
from acp import schema
from prompt_toolkit.application import create_app_session
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.data_structures import Size
from prompt_toolkit.document import Document
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output.vt100 import Vt100_Output
from rich.console import Console

from isaac_client import cli
from isaac_client.client import TerminalClient
from isaac_client.display import CommandSource, SlashCompleter, Terminal
from isaac_client.history import SessionHistory
from isaac_client.repl import command_catalog


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
        commands: CommandSource | None = None,
        toolbar: Callable[[], str] | None = None,
        default: str | Document = "",
        multiline: bool = False,
    ) -> str:
        if toolbar and "Enter queues" in toolbar():
            return await asyncio.Future[str]()
        if toolbar:
            self.toolbars.append(toolbar())
        return next(self.prompts, "/quit")

    async def choose(
        self, label: str, choices: list[str], *, toolbar: Callable[[], str] | None = None, search: bool = False
    ) -> int | None:
        self.menus[label] = choices
        return next(self.selections, None)


async def wait_for_output(transcript: io.StringIO, text: str) -> None:
    while text not in transcript.getvalue():
        await asyncio.sleep(0.01)


async def test_config_approval_and_named_resume(
    command: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    terminal = ScriptedTerminal(
        ["/help", "/pick-model", "/config", "Demo prompt", "/plan"],
        [1, 1, 0, 1],
    )
    monkeypatch.setattr(cli, "Terminal", lambda: terminal)

    async with asyncio.timeout(15):
        assert await cli.run_client(command, tmp_path, []) == 0
        output = capsys.readouterr().out
        assert "/demo  Run the demo" in output
        assert "/config  Change an advertised session setting" in output
        assert "/client:config  Change an advertised session setting" not in output
        assert "/pick-model  Choose the model offered by the agent" in output
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


async def test_preserves_streamed_lines_across_interactive_turns(
    command: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = io.StringIO()
    output = Vt100_Output(transcript, lambda: Size(rows=24, columns=100), enable_cpr=False)
    terminal = Terminal()
    terminal.interactive = True
    terminal.console = Console(file=transcript, force_terminal=True, color_system="standard")
    monkeypatch.setattr(cli, "Terminal", lambda: terminal)

    with create_pipe_input() as pipe, create_app_session(input=pipe, output=output):
        async with asyncio.timeout(15), asyncio.TaskGroup() as tasks:
            client = tasks.create_task(cli.run_client(command, tmp_path, []))
            await wait_for_output(transcript, "❯")
            pipe.send_text("Wait then end\n")
            await wait_for_output(transcript, "SDK Demo: Turn started")
            prefix = "\x1b[1mSDK Demo: \x1b[0m"
            await wait_for_output(transcript, f"{prefix}Turn startedTurn finished\n")
            pipe.send_text("Echo second\n")
            await wait_for_output(transcript, f"{prefix}Echo second\n")
            assert "Ctrl+D quit" in transcript.getvalue()
            pipe.send_text("\x04")
            assert await client == 0


def test_slash_palette_uses_agent_plan_and_help_and_keeps_model_picker() -> None:
    client = TerminalClient(Terminal())
    client.session.commands = [
        schema.AvailableCommand(name=name, description=name) for name in ("model", "plan", "help")
    ]
    completer = SlashCompleter(lambda: command_catalog(client).values())

    names = {item.text for item in completer.get_completions(Document("/"), CompleteEvent())}
    assert names == {"/model", "/pick-model", "/plan", "/help", "/config", "/quit"}
    commands = command_catalog(client)
    assert commands["/plan"].local is None
    assert commands["/help"].local is None
    assert commands["/pick-model"].local == "model"
    assert not list(completer.get_completions(Document("/client:"), CompleteEvent()))

    client.session.commands.append(schema.AvailableCommand(name="pick-model", description="Agent model picker"))
    commands = command_catalog(client)
    assert commands["/model"].local is None
    assert commands["/pick-model"].local is None
    assert all(command.local != "model" for command in commands.values())


async def test_slash_palette_and_multiline_composer(
    command: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = io.StringIO()
    output = Vt100_Output(transcript, lambda: Size(rows=24, columns=100), enable_cpr=False)
    terminal = Terminal()
    terminal.interactive = True
    terminal.console = Console(file=transcript, color_system=None)
    monkeypatch.setattr(cli, "Terminal", lambda: terminal)

    with create_pipe_input() as pipe, create_app_session(input=pipe, output=output):
        async with asyncio.timeout(8), asyncio.TaskGroup() as tasks:
            client = tasks.create_task(cli.run_client(command, tmp_path, []))
            await wait_for_output(transcript, "❯")
            pipe.send_text("/mo")
            await wait_for_output(transcript, "[model ID] Select an agent model")
            pipe.send_text("\x1b[B\r")
            await wait_for_output(transcript, "/model ")
            pipe.send_text("two\r")
            await wait_for_output(transcript, "SDK Demo: /model two\n")
            pipe.send_text("Echo first\x1b\r  second\r")
            await wait_for_output(transcript, "SDK Demo: Echo first\n")
            await wait_for_output(transcript, "  second\n")
            pipe.send_text("\x1b[200~Echo pasted\n  indentation\x1b[201~")
            await wait_for_output(transcript, "indentation")
            assert "SDK Demo: Echo pasted" not in transcript.getvalue()
            pipe.send_text("\r")
            await wait_for_output(transcript, "SDK Demo: Echo pasted\n")
            await wait_for_output(transcript, "  indentation\n")
            pipe.send_text("/quit\r")
            assert await client == 0


async def test_picker_keeps_multiline_draft_and_cursor() -> None:
    transcript = io.StringIO()
    output = Vt100_Output(transcript, lambda: Size(rows=24, columns=100), enable_cpr=False)
    terminal = Terminal()
    terminal.interactive = True

    with create_pipe_input() as pipe, create_app_session(input=pipe, output=output):
        async with asyncio.timeout(8), asyncio.TaskGroup() as tasks:
            editor = tasks.create_task(terminal.edit("❯ ", commands=None, toolbar=lambda: ""))
            await wait_for_output(transcript, "❯")
            pipe.send_text("First line\x1b\rsecond\x1b[D")
            await wait_for_output(transcript, "second")
            chooser = tasks.create_task(terminal.choose("Permission", ["Allow", "Reject"]))
            await wait_for_output(transcript, "2. Reject")
            pipe.send_text("\x1b[B\x1b[B\r")
            assert await chooser == 1
            pipe.send_text("X\r")
            assert await editor == "First line\nseconXd"


async def test_resume_from_composer(command: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    transcript = io.StringIO()
    output = Vt100_Output(transcript, lambda: Size(rows=24, columns=100), enable_cpr=False)
    terminal = Terminal()
    terminal.interactive = True
    terminal.console = Console(file=transcript, color_system=None)
    monkeypatch.setattr(cli, "Terminal", lambda: terminal)
    agent_command = [*command, "--list-sessions"]
    SessionHistory(agent_command, tmp_path).remember("missing", "Stale session")

    with create_pipe_input() as pipe, create_app_session(input=pipe, output=output):
        async with asyncio.timeout(8), asyncio.TaskGroup() as tasks:
            client = tasks.create_task(cli.run_client(agent_command, tmp_path, []))
            await wait_for_output(transcript, "❯")
            pipe.send_text("/resume\r")
            await wait_for_output(transcript, "Stale session")
            pipe.send_text("STALE\r")
            await wait_for_output(transcript, "Invalid params")
            pipe.send_text("Report session\r")
            await wait_for_output(transcript, "Current session is demo.\n")
            pipe.send_text("/resume\r")
            await wait_for_output(transcript, "Old session")
            pipe.send_text("OLD\r")
            await wait_for_output(transcript, "Session remote-old\n")
            assert "Session history replayed over ACP." in transcript.getvalue()
            pipe.send_text("Report session\r")
            await wait_for_output(transcript, "Current session is remote-old.\n")
            pipe.send_text("/quit\r")
            assert await client == 0


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
            editor = asyncio.create_task(terminal.edit("Queue ❯ ", commands=None, toolbar=lambda: ""))
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
            commands: CommandSource | None = None,
            toolbar: Callable[[], str] | None = None,
            default: str | Document = "",
            multiline: bool = False,
        ) -> str:
            nonlocal seen_input
            if toolbar and "Enter queues" in toolbar() and not seen_input:
                await started.wait()
                seen_input = True
                if cancel:
                    raise KeyboardInterrupt
                return "Echo queued follow-up"
            return await super().read(label, commands=commands, toolbar=toolbar, default=default, multiline=multiline)

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
