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
        busy: bool = False,
    ) -> str:
        if busy:
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


@pytest.mark.parametrize("choice", [0, 1, None])
async def test_proposal_picker_only_implements_when_selected(
    command: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    choice: int | None,
) -> None:
    terminal = ScriptedTerminal(["Propose a plan", "Echo next"], [] if choice is None else [choice])
    monkeypatch.setattr(cli, "Terminal", lambda: terminal)
    async with asyncio.timeout(8):
        assert await cli.run_client(command, tmp_path, []) == 0
    assert terminal.menus == {"Implement this plan?": ["Implement this plan", "Keep planning"]}
    output = capsys.readouterr().out
    assert "Update the example." in output
    assert ("Implementation began." in output) == (choice == 0)
    if choice == 0:
        assert output.index("Update the example.") < output.index("Implementation began.") < output.index("Echo next")


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
            assert "Working…" not in transcript.getvalue()
            pipe.send_text("Wait then end\n")
            await wait_for_output(transcript, "Working…")
            await wait_for_output(transcript, "SDK Demo: Turn started")
            prefix = "\x1b[1mSDK Demo: \x1b[0m"
            await wait_for_output(transcript, f"{prefix}Turn startedTurn finished\n")
            pipe.send_text("Echo second\n")
            await wait_for_output(transcript, f"{prefix}Echo second\n")
            assert "Ctrl+D quit" in transcript.getvalue()
            pipe.send_text("\x04")
            assert await client == 0


async def test_busy_composer_signals_activity_without_periodic_output() -> None:
    transcript = io.StringIO()
    output = Vt100_Output(transcript, lambda: Size(rows=24, columns=100), enable_cpr=False)
    terminal = Terminal()
    terminal.interactive = True
    client = TerminalClient(terminal)
    client.session.session_id = "demo"
    client.start_turn("Wait")

    with create_pipe_input() as pipe, create_app_session(input=pipe, output=output):
        async with asyncio.timeout(5), asyncio.TaskGroup() as tasks:
            editor = tasks.create_task(terminal.edit("❯ ", commands=None, toolbar=client.toolbar, busy=True))
            await wait_for_output(transcript, "\x1b[5 q")
            baseline = transcript.getvalue()
            assert "Working…" in baseline
            await asyncio.sleep(0.35)
            assert transcript.getvalue() == baseline
            await client.session_update("demo", schema.UsageUpdate(session_update="usage_update", used=25, size=1000))
            await wait_for_output(transcript, "Context 25/1,000")
            pipe.send_text("queued\r")
            assert await editor == "queued"
            assert "\x1b[0 q" in transcript.getvalue()[len(baseline) :]

            transcript.seek(0)
            transcript.truncate()
            idle = tasks.create_task(terminal.edit("❯ ", commands=None, toolbar=client.toolbar))
            await wait_for_output(transcript, "❯")
            assert "Working…" not in transcript.getvalue()
            assert "\x1b[5 q" not in transcript.getvalue()
            pipe.send_text("idle\r")
            assert await idle == "idle"


async def test_export_includes_complete_acp_content_and_stays_with_the_session(
    command: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    terminal = ScriptedTerminal(
        [
            "Show tool activity",
            "Demo prompt",
            "Echo first\n  second",
            "/export first session.txt",
            "/export first session.txt",
            "/new",
            "Echo another session",
            "/export second.txt",
            "/resume",
            "/export replay.txt",
        ],
        [0, 0],
    )
    monkeypatch.setattr(cli, "Terminal", lambda: terminal)
    async with asyncio.timeout(10):
        assert await cli.run_client([*command, "--replay-user"], tmp_path, []) == 0

    first = (tmp_path / "first session.txt").read_text()
    assert "You: Show tool activity" in first
    assert "You: Echo first\n  second" in first
    assert "SDK Demo: Echo first\n  second" in first
    assert "Thinking: Check the files." in first
    assert "\x1b" not in first and "[32m" not in first
    assert "Tool read-0: Read file" in first
    assert '"path": "demo.txt"' in first
    assert "Routine output" in first and "Text returned to the model" in first
    assert "Run tests (failed)" in first and "Test failed with details" in first
    assert "--- demo.txt\n+++ demo.txt" in first
    assert "\n-before\n" in first and "\n+after\n" in first
    assert "Could not export transcript" in capsys.readouterr().out
    second = (tmp_path / "second.txt").read_text()
    assert "You: Echo another session" in second
    assert "Read file" not in second
    replay = (tmp_path / "replay.txt").read_text()
    assert "You: Saved question" in replay
    assert "SDK Demo: Session history replayed over ACP." in replay
    assert "Echo another session" not in replay


async def test_model_preference_for_fresh_sessions_survives_restart_and_resume(
    command: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    command = [*command, "--model-preference"]
    history = SessionHistory(command, tmp_path)
    terminal = ScriptedTerminal(["/pick-model", "/new", "Echo new"], [1])
    monkeypatch.setattr(cli, "Terminal", lambda: terminal)
    async with asyncio.timeout(10):
        assert await cli.run_client(command, tmp_path, []) == 0
        assert history.preferred_model() == "two"
        assert any("Available models Model Two" in bar for bar in terminal.toolbars)

        terminal = ScriptedTerminal(["/new", "/model one", "/new", "Echo next"], [])
        assert await cli.run_client(command, tmp_path, [], session_id="remote-old") == 0
        assert "Available models Model One" in terminal.toolbars[0]  # From the saved session.
        assert "Available models Model Two" in terminal.toolbars[1]  # The preferred model survived the load.
        assert "Available models Model One" in terminal.toolbars[-1]
        assert history.preferred_model() == "one"

        history.prefer_model("unavailable")
        terminal = ScriptedTerminal(["Echo fallback"], [])
        assert await cli.run_client(command, tmp_path, []) == 0
        assert "Available models Model One" in terminal.toolbars[0]


def test_slash_palette_uses_agent_plan_and_help_and_keeps_model_picker() -> None:
    client = TerminalClient(Terminal())
    client.session.commands = [
        schema.AvailableCommand(name=name, description=name) for name in ("model", "plan", "help")
    ]
    completer = SlashCompleter(lambda: command_catalog(client).values())

    names = {item.text for item in completer.get_completions(Document("/"), CompleteEvent())}
    assert names == {"/model", "/pick-model", "/plan", "/new", "/export ", "/help", "/config", "/quit"}
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


async def test_picker_accepts_plain_terminal_input(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TERM", "dumb")
    transcript = io.StringIO()
    output = Vt100_Output(transcript, lambda: Size(rows=24, columns=100), enable_cpr=False)
    terminal = Terminal()
    terminal.interactive = True

    with create_pipe_input() as pipe, create_app_session(input=pipe, output=output):
        async with asyncio.timeout(8), asyncio.TaskGroup() as tasks:
            chooser = tasks.create_task(
                terminal.choose("Resume session", ["First session", "Second session"], search=True)
            )
            await wait_for_output(transcript, "2. Second session")
            pipe.send_text("2\r")
            assert await chooser == 1


async def test_session_mode_updates_without_advertised_choices() -> None:
    client = TerminalClient(Terminal())

    async def create() -> schema.NewSessionResponse:
        await client.session_update(
            "planning", schema.CurrentModeUpdate(session_update="current_mode_update", current_mode_id="plan")
        )
        return schema.NewSessionResponse(session_id="planning")

    await client.create_session(create)
    assert "Mode plan" in client.toolbar()
    await client.session_update(
        "planning", schema.CurrentModeUpdate(session_update="current_mode_update", current_mode_id="code")
    )
    assert "Mode code" in client.toolbar()


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


@pytest.mark.parametrize("close_supported", [False, True])
async def test_new_sessions_and_close_on_switch(
    command: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    close_supported: bool,
) -> None:
    log = tmp_path / "closed-sessions.txt"
    agent_command = [*command, "--fail-second-new", "--close-log", str(log)]
    if close_supported:
        agent_command.append("--advertise-close")
    terminal = ScriptedTerminal(
        [
            "Demo prompt",
            "/new",
            "Report session",
            "/new",
            "/help",
            "/plan",
            "Report session",
            "/resume",
            "Report session",
        ],
        [1, 0],
    )
    monkeypatch.setattr(cli, "Terminal", lambda: terminal)
    async with asyncio.timeout(15):
        assert await cli.run_client(agent_command, tmp_path, []) == 0

    output = capsys.readouterr().out
    assert "Could not start the new session." in output
    assert "Unopened session." not in output
    assert output.count("Current session is demo.") == 2
    assert "Session demo-3\n" in output and "Current session is demo-3." in output
    assert "/demo-new  Run the new session demo" in output
    assert "/demo  Run the demo" not in output
    assert "No plan items." in output
    assert "Session history replayed over ACP." in output
    if close_supported:
        assert log.read_text().splitlines() == ["demo", "demo-3", "demo"]
    else:
        assert not log.exists()
    assert {entry.session_id for entry in SessionHistory(agent_command, tmp_path).recent()} == {"demo", "demo-3"}


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
            busy: bool = False,
        ) -> str:
            nonlocal seen_input
            if busy and not seen_input:
                await started.wait()
                seen_input = True
                if cancel:
                    raise KeyboardInterrupt
                return "Echo queued follow-up"
            return await super().read(
                label, commands=commands, toolbar=toolbar, default=default, multiline=multiline, busy=busy
            )

    prompts = ["Wait for cancel", "Echo after cancellation"] if cancel else ["Wait then end"]
    terminal = BusyTerminal(prompts, [])
    monkeypatch.setattr(cli, "Terminal", lambda: terminal)
    async with asyncio.timeout(15):
        assert await cli.run_client([*command, "--plan-picker"], tmp_path, []) == 0
    assert "Implement this plan?" not in terminal.menus
    output = capsys.readouterr().out
    if cancel:
        assert "Stopped: cancelled" in output
        assert "Echo after cancellation" in output
    else:
        assert "Queued follow-up 1." in output
        assert output.index("Turn finished") < output.index("Echo queued follow-up")
