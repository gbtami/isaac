"""Exercise Isaac through the same stdio ACP transport used by Toad."""

import asyncio
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

import acp
import pytest
from acp import Client, schema
from acp.client.connection import ClientSideConnection


class RecordingClient:
    """A minimal SDK client for the capabilities exercised here."""

    def __init__(self) -> None:
        self.updates: list[tuple[str, object]] = []
        self.notifications: list[tuple[str, dict[str, object]]] = []
        self._update_received = asyncio.Event()
        self.approvals: list[schema.ToolCallUpdate] = []
        self.approval_kinds: list[set[str]] = []
        self.permission_started = asyncio.Event()
        self.permission_gate: asyncio.Event | None = None
        self.allow = True

    def on_connect(self, _connection: object) -> None:
        pass

    async def session_update(self, session_id: str, update: object, **_kwargs: object) -> None:
        self.updates.append((session_id, update))
        self._update_received.set()

    async def ext_notification(self, method: str, params: dict[str, object]) -> None:
        self.notifications.append((method, params))

    async def wait_for_update(self, session_id: str, matches: Callable[[object], bool], *, after: int = 0) -> None:
        while not any(received == session_id and matches(update) for received, update in self.updates[after:]):
            self._update_received.clear()
            await self._update_received.wait()

    async def request_permission(
        self,
        session_id: str,
        tool_call: schema.ToolCallUpdate,
        options: list[schema.PermissionOption],
        **_kwargs: object,
    ) -> schema.RequestPermissionResponse:
        self.approvals.append(tool_call)
        self.approval_kinds.append({option.kind for option in options})
        self.permission_started.set()
        if self.permission_gate is not None:
            await self.permission_gate.wait()
        kind = "allow_once" if self.allow else "reject_once"
        option = next(option for option in options if option.kind == kind)
        return schema.RequestPermissionResponse(
            outcome=schema.AllowedOutcome(outcome="selected", option_id=option.option_id)
        )


async def test_exhausted_output_budget_commits_tool_work_and_returns_a_normal_stop(tmp_path: Path) -> None:
    client = RecordingClient()
    fixture = Path(__file__).with_name("acp_fixture.py")
    async with (
        asyncio.timeout(30),
        acp.spawn_agent_process(
            cast(Client, client),
            sys.executable,
            str(fixture),
            "--exhaust-output",
            "--full-permissions",
            "--no-code-mode",
        ) as (connection, process),
    ):
        await connection.initialize(protocol_version=acp.PROTOCOL_VERSION)
        new = await connection.new_session(cwd=str(tmp_path), mcp_servers=[])
        prompt = json.dumps(
            {"tool": "write_file", "args": {"path": "record.txt", "content": "Completed successfully."}}
        )
        response = await connection.prompt(session_id=new.session_id, prompt=[acp.text_block(prompt)])
        assert response.stop_reason == "max_tokens"
        assert (tmp_path / "record.txt").read_text() == "Completed successfully."

        client.updates.clear()
        await connection.load_session(cwd=str(tmp_path), session_id=new.session_id, mcp_servers=[])
        await client.wait_for_update(
            new.session_id, lambda update: isinstance(update, schema.ToolCallProgress) and update.status == "completed"
        )
        resumed = await connection.prompt(session_id=new.session_id, prompt=[acp.text_block("/model")])
        assert resumed.stop_reason == "end_turn"
    assert process.stderr is not None
    assert b"Unhandled error" not in await process.stderr.read()


async def test_cli_initialization_and_model_selection(tmp_path: Path) -> None:
    """The installed entrypoint starts without provider credentials or a prompt."""
    client = RecordingClient()
    models = ("openai-codex:gpt-5.6-luna", "anthropic:claude-sonnet-4-6")
    async with (
        asyncio.timeout(30),
        acp.spawn_agent_process(
            cast(Client, client), sys.executable, "-m", "isaac", "--model", models[0], "--model", models[1]
        ) as (connection, _process),
    ):
        initialized = await connection.initialize(protocol_version=acp.PROTOCOL_VERSION)
        assert initialized.agent_info is not None
        assert initialized.agent_info.name == "isaac"
        assert initialized.agent_capabilities is not None
        assert initialized.agent_capabilities.mcp_capabilities is not None
        assert initialized.agent_capabilities.mcp_capabilities.http
        assert initialized.agent_capabilities.load_session
        session = await connection.new_session(cwd=str(tmp_path), mcp_servers=[])
        commands = [update for _, update in client.updates if isinstance(update, schema.AvailableCommandsUpdate)]
        assert {command.name for command in commands[-1].available_commands} == {
            "help",
            "plan",
            "checklist",
            "implement",
            "cancel-plan",
            "title",
            "compact",
            "model",
            "models",
            "permissions",
        }
        assert session.config_options is not None
        assert session.config_options[0].current_value == models[0]
        selected = await connection.set_config_option(session_id=session.session_id, config_id="model", value=models[1])
        assert selected.config_options[0].current_value == models[1]


@pytest.mark.parametrize("code_mode", [False, True])
async def test_plan_mode_restricts_tools_and_implements_the_reviewed_proposal(tmp_path: Path, code_mode: bool) -> None:
    fixture = Path(__file__).with_name("acp_fixture.py")
    args = [str(fixture), "--full-permissions"] + ([] if code_mode else ["--no-code-mode"])
    started = tmp_path / "mcp-started"
    mcp = schema.McpServerStdio(
        name="fixture", command=sys.executable, args=[str(fixture), "--mcp-fixture", str(started)], env=[]
    )
    evidence = tmp_path / "evidence.txt"
    evidence.write_text("Repository evidence.")

    async def ask(connection: ClientSideConnection, client: RecordingClient, session_id: str, text: str) -> str:
        start = len(client.updates)
        try:
            result = await connection.prompt(session_id=session_id, prompt=[acp.text_block(text)])
        except acp.RequestError as exc:
            raise AssertionError(f"{text}: {exc} {exc.data}") from exc
        assert result.stop_reason == "end_turn"
        await client.wait_for_update(
            session_id, lambda update: isinstance(update, schema.AgentMessageChunk), after=start
        )
        return "".join(
            update.content.text
            for received, update in client.updates[start:]
            if received == session_id
            and isinstance(update, schema.AgentMessageChunk)
            and isinstance(update.content, schema.TextContentBlock)
        )

    def tool(name: str, **arguments: object) -> str:
        return json.dumps({"tool": name, "args": arguments})

    original = RecordingClient()
    async with (
        asyncio.timeout(35),
        acp.spawn_agent_process(cast(Client, original), sys.executable, *args) as (connection, _process),
    ):
        await connection.initialize(
            protocol_version=acp.PROTOCOL_VERSION,
            client_capabilities=schema.ClientCapabilities(field_meta={"isaac/plan-proposed": True})
            if code_mode
            else None,
        )
        session = await connection.new_session(cwd=str(tmp_path), mcp_servers=[mcp])
        other = await connection.new_session(cwd=str(tmp_path), mcp_servers=[])
        sid = session.session_id
        assert "Create task.txt containing the reviewed text." in await ask(
            connection, original, sid, "/plan fixture proposal"
        )
        assert "Plan mode is inactive" in await ask(connection, original, other.session_id, "/implement")
        assert "Usage: /implement" in await ask(connection, original, sid, "/implement an unrelated task")
        assert "Use /implement" in await ask(connection, original, sid, "Go ahead and implement it.")
        await original.wait_for_update(
            sid, lambda update: isinstance(update, schema.CurrentModeUpdate) and update.current_mode_id == "plan"
        )
        assert await ask(connection, original, sid, tool("fixture_tool_catalog")) == "grep, list_files, read_file"
        assert "Repository evidence." in await ask(connection, original, sid, tool("read_file", path="evidence.txt"))
        for command in (
            tool("write_file", path="task.txt", content="unreviewed"),
            tool("shell", command="touch task.txt"),
            tool("run_code", code="write_file(path='task.txt', content='unreviewed')"),
            tool("delegate_task", agent_name="self", task="Create task.txt"),
            tool("fixture_hello"),
        ):
            assert "Unknown tool name:" in await ask(connection, original, sid, command)
        assert not original.approvals
        assert not started.exists() and not (tmp_path / "task.txt").exists()
        assert original.notifications == ([("isaac/plan-proposed", {"sessionId": sid})] if code_mode else [])

    resumed = RecordingClient()
    async with (
        asyncio.timeout(35),
        acp.spawn_agent_process(cast(Client, resumed), sys.executable, *args) as (connection, _process),
    ):
        await connection.initialize(protocol_version=acp.PROTOCOL_VERSION)
        await connection.load_session(cwd=str(tmp_path), session_id=sid, mcp_servers=[mcp])
        await resumed.wait_for_update(
            sid, lambda update: isinstance(update, schema.CurrentModeUpdate) and update.current_mode_id == "plan"
        )
        await resumed.wait_for_update(
            sid,
            lambda update: (
                isinstance(update, schema.AgentMessageChunk)
                and isinstance(update.content, schema.TextContentBlock)
                and "Unknown tool name: 'fixture_hello'" in update.content.text
            ),
        )
        assert await ask(connection, resumed, sid, tool("fixture_tool_catalog")) == "grep, list_files, read_file"
        assert not started.exists()
        await ask(connection, resumed, sid, "/implement")
        assert (tmp_path / "task.txt").read_text() == "reviewed"
        assert started.exists()
        await resumed.wait_for_update(
            sid, lambda update: isinstance(update, schema.CurrentModeUpdate) and update.current_mode_id == "code"
        )
        assert "MCP is available." in await ask(connection, resumed, sid, tool("fixture_hello"))
        assert "Plan mode is on" in await ask(connection, resumed, sid, "/plan")
        assert "Plan mode is off" in await ask(connection, resumed, sid, "/cancel-plan")
        assert "Plan mode is inactive" in await ask(connection, resumed, sid, "/implement")


@pytest.mark.parametrize(("code_mode", "planning"), [(False, False), (True, False), (True, True)])
async def test_harness_questions_round_trip_over_acp(tmp_path: Path, code_mode: bool, planning: bool) -> None:
    class AnsweringClient(RecordingClient):
        async def create_elicitation(
            self, message: str, mode: schema.ElicitationMode, **_kwargs: object
        ) -> schema.CreateElicitationResponse:
            assert isinstance(mode, schema.ElicitationFormSessionMode)
            assert mode.requested_schema.properties is not None
            options = mode.requested_schema.properties["choice_0"]
            assert isinstance(options, schema.ElicitationMultiSelectPropertySchema)
            assert isinstance(options.items, schema.TitledMultiSelectItems)
            assert [option.const for option in options.items.any_of] == ["Unit tests", "Type checks"]
            return schema.AcceptElicitationResponse(
                action="accept", content={"choice_0": ["Unit tests", "Type checks"], "other_1": "Python 3.13"}
            )

    fixture = Path(__file__).with_name("acp_fixture.py")
    client = AnsweringClient()
    questions = [
        {
            "header": "Checks",
            "question": "Which checks should run?",
            "options": [{"label": "Unit tests"}, {"label": "Type checks"}],
            "multi_select": True,
        },
        {
            "header": "Version",
            "question": "Which Python version should we target?",
            "options": [{"label": "Python 3.11"}, {"label": "Python 3.12"}],
        },
    ]
    command = (
        {"tool": "run_code", "args": {"code": f"print(await ask_user_question(questions={questions!r}))"}}
        if code_mode and not planning
        else {"tool": "ask_user_question", "args": {"questions": questions}}
    )
    args = [str(fixture)] + ([] if code_mode else ["--no-code-mode"])
    async with (
        asyncio.timeout(30),
        acp.spawn_agent_process(cast(Client, client), sys.executable, *args, use_unstable_protocol=True) as (
            connection,
            _process,
        ),
    ):
        await connection.initialize(
            protocol_version=acp.PROTOCOL_VERSION,
            client_capabilities=schema.ClientCapabilities(
                elicitation=schema.ElicitationCapabilities(form=schema.ElicitationFormCapabilities())
            ),
        )
        session = await connection.new_session(cwd=str(tmp_path), mcp_servers=[])
        if planning:
            await connection.prompt(session_id=session.session_id, prompt=[acp.text_block("/plan")])
        result = await connection.prompt(session_id=session.session_id, prompt=[acp.text_block(json.dumps(command))])
        assert result.stop_reason == "end_turn"
        text = "".join(
            update.content.text
            for _, update in client.updates
            if isinstance(update, schema.AgentMessageChunk) and isinstance(update.content, schema.TextContentBlock)
        )
        assert "Unit tests" in text and "Type checks" in text and "Python 3.13" in text


async def test_context_usage_is_reported_over_acp(tmp_path: Path) -> None:
    client = RecordingClient()
    fixture = Path(__file__).with_name("acp_fixture.py")
    async with (
        asyncio.timeout(30),
        acp.spawn_agent_process(cast(Client, client), sys.executable, str(fixture), "--known-context-window") as (
            connection,
            _process,
        ),
    ):
        await connection.initialize(protocol_version=acp.PROTOCOL_VERSION)
        session = await connection.new_session(cwd=str(tmp_path), mcp_servers=[])
        result = await connection.prompt(session_id=session.session_id, prompt=[acp.text_block("Check this project.")])
        assert result.stop_reason == "end_turn"
        readings = [
            update
            for sid, update in client.updates
            if sid == session.session_id and isinstance(update, schema.UsageUpdate)
        ]
        assert readings
        assert readings[-1].size == 50_000
        assert 0 < readings[-1].used < readings[-1].size


async def test_background_delegate_can_be_listed_steered_and_cancelled(tmp_path: Path) -> None:
    """Manage a real Harness self-delegate through the default Code Mode tool."""
    fixture = Path(__file__).with_name("acp_fixture.py")
    client = RecordingClient()
    code = (
        "receipt = await delegate_task(agent_name='self', task='fixture block until cancelled', background=True)\n"
        "print(receipt)\n"
        "task_id = receipt.split()[1]\n"
        "print(await list_tasks())\n"
        "print(await message_task(task_id=task_id, message='Check another file.'))\n"
        "print(await cancel_task(task_id=task_id, force=True))\n"
        "print(await list_tasks())"
    )
    async with (
        asyncio.timeout(30),
        acp.spawn_agent_process(cast(Client, client), sys.executable, str(fixture)) as (connection, _process),
    ):
        await connection.initialize(protocol_version=acp.PROTOCOL_VERSION)
        session = await connection.new_session(cwd=str(tmp_path), mcp_servers=[])
        prompt = acp.text_block(json.dumps({"tool": "run_code", "args": {"code": code}}))
        assert (await connection.prompt(session_id=session.session_id, prompt=[prompt])).stop_reason == "end_turn"
        output = "".join(
            update.content.text
            for _, update in client.updates
            if isinstance(update, schema.AgentMessageChunk) and isinstance(update.content, schema.TextContentBlock)
        )
        assert "running in the background" in output
        assert "self, running, background" in output
        assert "Message sent to task" in output
        assert "self, finished: cancelled" in output


async def test_named_delegate_loads_from_session_workspace(tmp_path: Path) -> None:
    fixture = Path(__file__).with_name("acp_fixture.py")
    client = RecordingClient()
    agents = tmp_path / ".agents" / "agents"
    agents.mkdir(parents=True)
    (agents / "reviewer.md").write_text(
        "---\nname: reviewer\ndescription: A fixture review delegate.\n---\nfixture_named_agent_instructions\n",
        encoding="utf-8",
    )
    async with (
        asyncio.timeout(30),
        acp.spawn_agent_process(
            cast(Client, client), sys.executable, str(fixture), "--named-delegates", "--no-code-mode"
        ) as (
            connection,
            _process,
        ),
    ):
        await connection.initialize(protocol_version=acp.PROTOCOL_VERSION)
        session = await connection.new_session(cwd=str(tmp_path), mcp_servers=[])
        prompt = acp.text_block(
            json.dumps({"tool": "delegate_task", "args": {"agent_name": "reviewer", "task": "Review this change."}})
        )
        assert (await connection.prompt(session_id=session.session_id, prompt=[prompt])).stop_reason == "end_turn"
        output = "".join(
            update.content.text
            for _, update in client.updates
            if isinstance(update, schema.AgentMessageChunk) and isinstance(update.content, schema.TextContentBlock)
        )
        assert "Named delegate used" in output


async def test_model_commands_share_acp_selection_and_resume(tmp_path: Path) -> None:
    fixture = Path(__file__).with_name("acp_fixture.py")
    inspect = json.dumps({"tool": "model_name", "args": {}})

    async def prompt(connection: ClientSideConnection, client: RecordingClient, session_id: str, text: str) -> str:
        start = len(client.updates)
        response = await connection.prompt(session_id=session_id, prompt=[acp.text_block(text)])
        assert response.stop_reason == "end_turn"
        return "".join(
            update.content.text
            for received_session, update in client.updates[start:]
            if received_session == session_id
            and isinstance(update, schema.AgentMessageChunk)
            and isinstance(update.content, schema.TextContentBlock)
        )

    client = RecordingClient()
    async with (
        asyncio.timeout(30),
        acp.spawn_agent_process(cast(Client, client), sys.executable, str(fixture)) as (connection, _process),
    ):
        await connection.initialize(protocol_version=acp.PROTOCOL_VERSION)
        session = await connection.new_session(cwd=str(tmp_path), mcp_servers=[])

        async def send(text: str) -> str:
            return await prompt(connection, client, session.session_id, text)

        assert await send("/models") == "- fixture:primary (current)\n- fixture:secondary"
        assert "Unknown model" in await send("/model fixture:unknown")
        assert await send(inspect) == "fixture:primary"
        assert "Model set to fixture:secondary" in await send("/model fixture:secondary")
        assert await send(inspect) == "fixture:secondary"
        updates = [update for _, update in client.updates if isinstance(update, schema.ConfigOptionUpdate)]
        assert updates[-1].config_options[0].current_value == "fixture:secondary"
        await connection.set_config_option(session_id=session.session_id, config_id="model", value="fixture:primary")
        assert "Current model is fixture:primary" in await send("/model")
        await send("/model fixture:secondary")
        await send("/permissions full")

    resumed = RecordingClient()
    async with (
        asyncio.timeout(30),
        acp.spawn_agent_process(cast(Client, resumed), sys.executable, str(fixture)) as (connection, _process),
    ):
        await connection.initialize(protocol_version=acp.PROTOCOL_VERSION)
        loaded = await connection.load_session(session_id=session.session_id, cwd=str(tmp_path), mcp_servers=[])
        assert loaded is not None and loaded.config_options is not None
        assert loaded.config_options[0].current_value == "fixture:secondary"
        assert (await prompt(connection, resumed, session.session_id, inspect)).endswith("fixture:secondary")
        assert "Permission mode is read-only" in await prompt(connection, resumed, session.session_id, "/permissions")


async def test_permissions_command_changes_only_its_live_session(tmp_path: Path) -> None:
    fixture = Path(__file__).with_name("acp_fixture.py")
    client = RecordingClient()
    client.allow = False
    async with (
        asyncio.timeout(30),
        acp.spawn_agent_process(cast(Client, client), sys.executable, str(fixture)) as (connection, _process),
    ):
        await connection.initialize(protocol_version=acp.PROTOCOL_VERSION)
        first, second = [await connection.new_session(cwd=str(tmp_path), mcp_servers=[]) for _ in range(2)]

        async def send(session_id: str, text: str) -> None:
            response = await connection.prompt(session_id=session_id, prompt=[acp.text_block(text)])
            assert response.stop_reason == "end_turn"

        async def write(session_id: str, filename: str) -> None:
            await send(session_id, json.dumps({"tool": "write_file", "args": {"path": filename, "content": "sample"}}))

        await send(first.session_id, "/permissions full")
        await write(first.session_id, "full.txt")
        assert (tmp_path / "full.txt").read_text() == "sample"
        assert not client.approvals

        await write(second.session_id, "denied.txt")
        assert not (tmp_path / "denied.txt").exists()
        assert len(client.approvals) == 1

        await send(first.session_id, "/permissions read-only")
        await send(first.session_id, "/permissions invalid")
        await write(first.session_id, "read-only.txt")
        assert not (tmp_path / "read-only.txt").exists()
        assert len(client.approvals) == 2


async def test_workspace_tools_and_permission_denial_over_stdio(tmp_path: Path) -> None:
    """Two workspaces, real Harness tools, and an ACP denial before any write."""
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "note.txt").write_text("first workspace", encoding="utf-8")
    (second / "note.txt").write_text("second workspace", encoding="utf-8")
    client = RecordingClient()
    fixture = Path(__file__).with_name("acp_fixture.py")
    async with (
        asyncio.timeout(30),
        acp.spawn_agent_process(cast(Client, client), sys.executable, str(fixture), "--no-code-mode") as (
            connection,
            _process,
        ),
    ):
        await connection.initialize(protocol_version=acp.PROTOCOL_VERSION)
        sessions = [await connection.new_session(cwd=str(path), mcp_servers=[]) for path in (first, second)]

        async def prompt(session_id: str, tool: str, args: dict[str, object]) -> str:
            start = len(client.updates)
            result = await connection.prompt(
                session_id=session_id, prompt=[acp.text_block(json.dumps({"tool": tool, "args": args}))]
            )
            assert result.stop_reason == "end_turn"
            return "".join(
                update.content.text
                for received_session, update in client.updates[start:]
                if received_session == session_id
                and isinstance(update, schema.AgentMessageChunk)
                and isinstance(update.content, schema.TextContentBlock)
            )

        for session, expected in zip(sessions, ("first workspace", "second workspace"), strict=True):
            assert expected in await prompt(session.session_id, "read_file", {"path": "note.txt"})

        await prompt(sessions[0].session_id, "write_file", {"path": "approved.txt", "content": "approved"})
        assert (first / "approved.txt").read_text() == "approved"
        assert not (second / "approved.txt").exists()
        assert len(client.approvals) == 1

        client.allow = False
        await prompt(sessions[1].session_id, "write_file", {"path": "denied.txt", "content": "denied"})
        assert not (second / "denied.txt").exists()
        assert len(client.approvals) == 2

        client.allow = True
        output = await prompt(sessions[1].session_id, "shell", {"command": "pwd"})
        assert str(second) in output
        assert len(client.approvals) == 3


async def test_code_mode_approvals_over_stdio(tmp_path: Path) -> None:
    """The nested write waits for ACP, and denial or cancellation leaves it unexecuted."""
    client = RecordingClient()
    fixture = Path(__file__).with_name("acp_fixture.py")
    async with (
        asyncio.timeout(30),
        acp.spawn_agent_process(cast(Client, client), sys.executable, str(fixture)) as (connection, _process),
    ):
        await connection.initialize(protocol_version=acp.PROTOCOL_VERSION)
        session = await connection.new_session(cwd=str(tmp_path), mcp_servers=[])

        async def write(name: str) -> schema.PromptResponse:
            code = f"print(write_file(path={name!r}, content='sample'))"
            prompt = acp.text_block(json.dumps({"tool": "run_code", "args": {"code": code}}))
            return await connection.prompt(session_id=session.session_id, prompt=[prompt])

        client.permission_gate = asyncio.Event()
        pending = asyncio.create_task(write("approved.txt"))
        await client.permission_started.wait()
        assert not (tmp_path / "approved.txt").exists()
        assert client.approval_kinds[-1] == {"allow_once", "reject_once"}
        approved = client.approvals[-1]
        assert approved.title == "Write approved.txt"
        assert approved.content and isinstance(diff := approved.content[0], schema.FileEditToolCallContent)
        assert diff.path == str(tmp_path / "approved.txt") and diff.new_text == "sample"
        client.permission_gate.set()
        assert (await pending).stop_reason == "end_turn"
        assert (tmp_path / "approved.txt").read_text() == "sample"

        client.permission_gate = None
        client.allow = False
        assert (await write("denied.txt")).stop_reason == "end_turn"
        assert len(client.approvals) == 2
        assert not (tmp_path / "denied.txt").exists()

        client.allow = True
        client.permission_started.clear()
        client.permission_gate = asyncio.Event()
        pending = asyncio.create_task(write("cancelled.txt"))
        await client.permission_started.wait()
        await connection.cancel(session_id=session.session_id)
        assert (await pending).stop_reason == "cancelled"
        # A late approval for the cancelled call must not execute it.
        client.permission_gate.set()
        assert (await write("after-cancel.txt")).stop_reason == "end_turn"
        assert (tmp_path / "after-cancel.txt").is_file()
        assert not (tmp_path / "cancelled.txt").exists()

        shell = acp.text_block(json.dumps({"tool": "shell", "args": {"command": "pwd"}}))
        assert (await connection.prompt(session_id=session.session_id, prompt=[shell])).stop_reason == "end_turn"
        assert client.approval_kinds[-1] == {"allow_once", "reject_once", "allow_always", "reject_always"}


@pytest.mark.parametrize("code_mode", [False, True])
async def test_plan_sidebar_and_commands(tmp_path: Path, code_mode: bool) -> None:
    """Native plan snapshots follow whole-plan edits across turns, including CodeMode."""
    client = RecordingClient()
    fixture = Path(__file__).with_name("acp_fixture.py")
    args = [str(fixture), *([] if code_mode else ["--no-code-mode"])]
    async with (
        asyncio.timeout(30),
        acp.spawn_agent_process(cast(Client, client), sys.executable, *args) as (connection, _process),
    ):
        await connection.initialize(protocol_version=acp.PROTOCOL_VERSION)
        session = await connection.new_session(cwd=str(tmp_path), mcp_servers=[])
        other = await connection.new_session(cwd=str(tmp_path), mcp_servers=[])

        async def prompt(text: str, *, session_id: str = session.session_id) -> str:
            start = len(client.updates)
            response = await connection.prompt(session_id=session_id, prompt=[acp.text_block(text)])
            assert response.stop_reason == "end_turn"
            return "".join(
                update.content.text
                for received, update in client.updates[start:]
                if received == session_id
                and isinstance(update, schema.AgentMessageChunk)
                and isinstance(update.content, schema.TextContentBlock)
            )

        def plans() -> list[schema.AgentPlanUpdate]:
            return [
                update
                for received, update in client.updates
                if received == session.session_id and isinstance(update, schema.AgentPlanUpdate)
            ]

        assert "empty" in await prompt("/compact")
        assert "/compact" in await prompt("/help")
        assert "The checklist is empty" in await prompt("/checklist")
        assert "Usage: /checklist" in await prompt("/checklist unexpected-argument")
        assert "ready" == await prompt("/unknown-command")
        assert "ready" == await prompt("/help\nPlease explain this text.")
        items = [
            {"id": "inspect", "content": "Inspect code", "status": "in_progress"},
            {"id": "fix", "content": "Fix code", "status": "pending"},
        ]
        await prompt(json.dumps({"tool": "write_plan", "args": {"items": items}}))
        assert [entry.content for entry in plans()[-1].entries] == ["Inspect code", "Fix code"]
        assert [entry.status for entry in plans()[-1].entries] == ["in_progress", "pending"]
        assert "Inspect code" in await prompt("/checklist")
        assert "The checklist is empty" in await prompt("/checklist", session_id=other.session_id)

        await prompt(json.dumps({"tool": "write_plan", "args": {"items": list(reversed(items))}}))
        assert [entry.content for entry in plans()[-1].entries] == ["Fix code", "Inspect code"]
        items[0]["status"] = "completed"
        items[1]["status"] = "cancelled"
        await prompt(json.dumps({"tool": "write_plan", "args": {"items": items}}))
        assert [(entry.content, entry.status) for entry in plans()[-1].entries] == [("Inspect code", "completed")]
        await prompt(json.dumps({"tool": "write_plan", "args": {"items": []}}))
        assert plans()[-1].entries == []


@pytest.mark.parametrize("mode", ["focus", "fail-summary", "fail-native"])
async def test_compact_commits_history_for_next_turn(tmp_path: Path, mode: str) -> None:
    """The command uses Harness compaction and the adapter commits the resulting history."""
    client = RecordingClient()
    fixture = Path(__file__).with_name("acp_fixture.py")
    args = [str(fixture), "--native-compact"]
    if mode == "fail-summary":
        args.append("--fail-summary")
    if mode == "fail-native":
        args.append("--fail-native-compact")
    (tmp_path / "long.txt").write_text("abcdefghij " * 1000)
    async with (
        asyncio.timeout(30),
        acp.spawn_agent_process(cast(Client, client), sys.executable, *args) as (connection, _process),
    ):
        await connection.initialize(protocol_version=acp.PROTOCOL_VERSION)
        session = await connection.new_session(cwd=str(tmp_path), mcp_servers=[])

        async def prompt(text: str) -> str:
            start = len(client.updates)
            result = await connection.prompt(session_id=session.session_id, prompt=[acp.text_block(text)])
            assert result.stop_reason == "end_turn"
            return "".join(
                update.content.text
                for _, update in client.updates[start:]
                if isinstance(update, schema.AgentMessageChunk) and isinstance(update.content, schema.TextContentBlock)
            )

        assert "empty" in await prompt("/compact")
        await prompt("hello")
        for _ in range(5):
            await prompt(json.dumps({"tool": "read_file", "args": {"path": "long.txt"}}))
        inspect = json.dumps({"tool": "history_length", "args": {}})
        before = int(await prompt(inspect))
        command = "/compact" if mode == "fail-native" else "/compact preserve the current task"
        assert "tokens saved" in await prompt(command)
        after = int(await prompt(inspect))
        assert after < before


async def test_native_compaction_survives_process_restart(tmp_path: Path) -> None:
    fixture = Path(__file__).with_name("acp_fixture.py")
    args = [sys.executable, str(fixture), "--native-compact"]
    client = RecordingClient()
    async with asyncio.timeout(30), acp.spawn_agent_process(cast(Client, client), *args) as (connection, _process):
        await connection.initialize(protocol_version=acp.PROTOCOL_VERSION)
        session = await connection.new_session(cwd=str(tmp_path), mcp_servers=[])
        await connection.prompt(session_id=session.session_id, prompt=[acp.text_block("remember the build task")])
        response = await connection.prompt(session_id=session.session_id, prompt=[acp.text_block("/compact")])
        assert response.stop_reason == "end_turn"
        updates = [update for _, update in client.updates if isinstance(update, schema.AgentMessageChunk)]
        assert any(
            isinstance(update.content, schema.TextContentBlock) and "using native compaction" in update.content.text
            for update in updates
        )

    client = RecordingClient()
    async with asyncio.timeout(30), acp.spawn_agent_process(cast(Client, client), *args) as (connection, _process):
        await connection.initialize(protocol_version=acp.PROTOCOL_VERSION)
        await connection.load_session(session_id=session.session_id, cwd=str(tmp_path), mcp_servers=[])
        check = json.dumps({"tool": "check_compaction", "args": {}})
        response = await connection.prompt(session_id=session.session_id, prompt=[acp.text_block(check)])
        assert response.stop_reason == "end_turn"
        assert any(
            isinstance(update, schema.AgentMessageChunk)
            and isinstance(update.content, schema.TextContentBlock)
            and "compaction restored" in update.content.text
            for _, update in client.updates
        )


@pytest.mark.parametrize(("has_completed_turn", "code_mode"), [(False, True), (True, False)])
async def test_session_load_recovers_checkpoint_after_process_kill(
    tmp_path: Path, has_completed_turn: bool, code_mode: bool
) -> None:
    fixture = Path(__file__).with_name("acp_fixture.py")
    effect = tmp_path / "partial-effect.txt"
    (tmp_path / "evidence.txt").write_text("checkpoint evidence")
    args = [sys.executable, str(fixture), "--checkpoint-probe", str(effect)]
    if not code_mode:
        args.append("--no-code-mode")

    async with asyncio.timeout(30):
        async with acp.spawn_agent_process(cast(Client, RecordingClient()), *args) as (connection, process):
            await connection.initialize(protocol_version=acp.PROTOCOL_VERSION)
            session = await connection.new_session(cwd=str(tmp_path), mcp_servers=[])
            session_id = session.session_id
            if has_completed_turn:
                await connection.prompt(session_id=session_id, prompt=[acp.text_block("Initial completed turn")])
            if code_mode:
                command = json.dumps(
                    {"tool": "run_code", "args": {"code": "print(await read_file(path='evidence.txt'))"}}
                )
            else:
                command = json.dumps({"tool": "read_file", "args": {"path": "evidence.txt"}})
            pending = asyncio.create_task(connection.prompt(session_id=session_id, prompt=[acp.text_block(command)]))
            try:
                while not effect.exists() or not effect.stat().st_size:
                    assert not pending.done(), "The agent finished before the probe reached its side effect"
                    await asyncio.sleep(0.01)
                process.kill()
                await process.wait()
            finally:
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)

        resumed = RecordingClient()
        async with acp.spawn_agent_process(cast(Client, resumed), *args) as (connection, _process):
            await connection.initialize(protocol_version=acp.PROTOCOL_VERSION)
            await connection.load_session(session_id=session_id, cwd=str(tmp_path), mcp_servers=[])
            assert effect.read_text() == "started\n"
            assert resumed.approvals == []
            response = await connection.prompt(
                session_id=session_id, prompt=[acp.text_block("fixture continue from checkpoint")]
            )
            assert response.stop_reason == "end_turn"
            assert effect.read_text() == "started\n"
            assert any(
                isinstance(update, schema.AgentMessageChunk)
                and isinstance(update.content, schema.TextContentBlock)
                and "Recovered a checkpoint" in update.content.text
                and ("run_code" if code_mode else "checkpoint_probe") in update.content.text
                for _, update in resumed.updates
            )
            assert any(
                isinstance(update, schema.AgentMessageChunk)
                and isinstance(update.content, schema.TextContentBlock)
                and "Restored evidence and closed the unfinished call." in update.content.text
                for _, update in resumed.updates
            )


async def test_session_load_after_process_restart(tmp_path: Path) -> None:
    """Autosave survives process exit; load replays history and restores model and plan, without tools running again."""
    fixture = Path(__file__).with_name("acp_fixture.py")
    client = RecordingClient()
    async with (
        asyncio.timeout(30),
        acp.spawn_agent_process(cast(Client, client), sys.executable, str(fixture)) as (connection, _process),
    ):
        initialized = await connection.initialize(protocol_version=acp.PROTOCOL_VERSION)
        assert initialized.agent_capabilities is not None and initialized.agent_capabilities.load_session
        session = await connection.new_session(cwd=str(tmp_path), mcp_servers=[])
        untouched = await connection.new_session(cwd=str(tmp_path), mcp_servers=[])
        for tool, arguments in (
            ("write_file", {"path": "saved.txt", "content": "original"}),
            ("write_plan", {"items": [{"id": "next", "content": "Continue the task", "status": "in_progress"}]}),
        ):
            response = await connection.prompt(
                session_id=session.session_id,
                prompt=[acp.text_block(json.dumps({"tool": tool, "args": arguments}))],
            )
            assert response.stop_reason == "end_turn"
        await connection.set_config_option(session_id=session.session_id, config_id="model", value="fixture:secondary")

    assert (tmp_path / "saved.txt").read_text() == "original"
    (tmp_path / "saved.txt").unlink()
    resumed = RecordingClient()
    async with (
        asyncio.timeout(30),
        acp.spawn_agent_process(cast(Client, resumed), sys.executable, str(fixture)) as (connection, _process),
    ):
        await connection.initialize(protocol_version=acp.PROTOCOL_VERSION)
        loaded = await connection.load_session(session_id=session.session_id, cwd=str(tmp_path), mcp_servers=[])
        assert loaded is not None and loaded.config_options is not None
        assert loaded.config_options[0].current_value == "fixture:secondary"
        assert not (tmp_path / "saved.txt").exists()
        assert resumed.approvals == []
        # The SDK dispatches replay notifications independently of the load response.
        await resumed.wait_for_update(
            session.session_id,
            lambda update: (
                isinstance(update, schema.UserMessageChunk)
                and isinstance(update.content, schema.TextContentBlock)
                and '"tool": "write_plan"' in update.content.text
            ),
        )
        assert any(isinstance(update, schema.UserMessageChunk) for _, update in resumed.updates)
        assert any(isinstance(update, schema.AgentMessageChunk) for _, update in resumed.updates)
        assert any(isinstance(update, schema.ToolCallStart) for _, update in resumed.updates)
        assert not any(
            isinstance(update, schema.AgentMessageChunk)
            and isinstance(update.content, schema.TextContentBlock)
            and "Recovered a checkpoint" in update.content.text
            for _, update in resumed.updates
        )
        plans = [update for _, update in resumed.updates if isinstance(update, schema.AgentPlanUpdate)]
        assert [(entry.content, entry.status) for entry in plans[-1].entries] == [("Continue the task", "in_progress")]

        start = len(resumed.updates)
        response = await connection.prompt(session_id=session.session_id, prompt=[acp.text_block("/checklist")])
        assert response.stop_reason == "end_turn"
        await resumed.wait_for_update(
            session.session_id,
            lambda update: (
                isinstance(update, schema.AgentMessageChunk)
                and isinstance(update.content, schema.TextContentBlock)
                and "Continue the task" in update.content.text
            ),
            after=start,
        )
        start = len(resumed.updates)
        response = await connection.prompt(
            session_id=session.session_id,
            prompt=[acp.text_block(json.dumps({"tool": "history_length", "args": {}}))],
        )
        assert response.stop_reason == "end_turn"
        await resumed.wait_for_update(
            session.session_id, lambda update: isinstance(update, schema.AgentMessageChunk), after=start
        )
        assert response.stop_reason == "end_turn"
        length = "".join(
            update.content.text
            for _, update in resumed.updates[start:]
            if isinstance(update, schema.AgentMessageChunk) and isinstance(update.content, schema.TextContentBlock)
        )
        assert int(length) > 2  # The model received the saved conversation, not just the new prompt.
        # An empty session is saved at creation too, and restoring it must not inherit the other plan/model.
        empty = await connection.load_session(session_id=untouched.session_id, cwd=str(tmp_path), mcp_servers=[])
        assert empty is not None and empty.config_options is not None
        assert empty.config_options[0].current_value == "fixture:primary"
        await resumed.wait_for_update(untouched.session_id, lambda update: isinstance(update, schema.AgentPlanUpdate))
        plans = [
            update
            for received, update in resumed.updates
            if received == untouched.session_id and isinstance(update, schema.AgentPlanUpdate)
        ]
        assert plans[-1].entries == []


async def test_stale_session_cannot_replace_a_newer_turn(tmp_path: Path) -> None:
    fixture = Path(__file__).with_name("acp_fixture.py")
    stale = RecordingClient()
    stale.allow = False
    stale.permission_gate = asyncio.Event()
    write = json.dumps({"tool": "write_file", "args": {"path": "stale.txt", "content": "stale turn"}})
    newer = "Keep the newer turn"
    reopened = RecordingClient()

    async with asyncio.timeout(30):
        async with (
            acp.spawn_agent_process(cast(Client, RecordingClient()), sys.executable, str(fixture)) as (first, _),
            acp.spawn_agent_process(cast(Client, stale), sys.executable, str(fixture)) as (second, _),
        ):
            for connection in (first, second):
                await connection.initialize(protocol_version=acp.PROTOCOL_VERSION)
            session_id = (await first.new_session(cwd=str(tmp_path), mcp_servers=[])).session_id
            await first.prompt(session_id=session_id, prompt=[acp.text_block("Initial shared turn")])
            await second.load_session(session_id=session_id, cwd=str(tmp_path), mcp_servers=[])
            async with asyncio.TaskGroup() as turns:
                pending = turns.create_task(second.prompt(session_id=session_id, prompt=[acp.text_block(write)]))
                await stale.permission_started.wait()
                await first.prompt(session_id=session_id, prompt=[acp.text_block(newer)])
                stale.permission_gate.set()
            assert pending.result().stop_reason == "end_turn"
            await stale.wait_for_update(
                session_id,
                lambda update: (
                    isinstance(update, schema.AgentMessageChunk)
                    and isinstance(update.content, schema.TextContentBlock)
                    and "Reopen the session" in update.content.text
                ),
            )
            with pytest.raises(acp.RequestError, match="Reopen the session"):
                await second.prompt(session_id=session_id, prompt=[acp.text_block("Another stale turn")])

        async with acp.spawn_agent_process(cast(Client, reopened), sys.executable, str(fixture)) as (connection, _):
            await connection.initialize(protocol_version=acp.PROTOCOL_VERSION)
            await connection.load_session(session_id=session_id, cwd=str(tmp_path), mcp_servers=[])
            response = await connection.prompt(
                session_id=session_id, prompt=[acp.text_block(json.dumps({"tool": "model_name", "args": {}}))]
            )
            assert response.stop_reason == "end_turn"
            await reopened.wait_for_update(
                session_id,
                lambda update: (
                    isinstance(update, schema.AgentMessageChunk)
                    and isinstance(update.content, schema.TextContentBlock)
                    and update.content.text == "fixture:primary"
                ),
            )
            replay = [
                update.content.text
                for _, update in reopened.updates
                if isinstance(update, (schema.UserMessageChunk, schema.AgentMessageChunk))
                and isinstance(update.content, schema.TextContentBlock)
            ]
            assert newer in replay
            assert write not in replay
            assert not any("Recovered a checkpoint" in text for text in replay)


@pytest.mark.parametrize("fail_title", [False, True])
async def test_automatic_session_title_and_resume(tmp_path: Path, fail_title: bool) -> None:
    """The first coding prompt gets a stable title; naming failures never fail the turn."""
    fixture = Path(__file__).with_name("acp_fixture.py")
    args = [str(fixture), *(["--fail-title"] if fail_title else [])]
    first_prompt = "Please improve session persistence so old conversations can resume"
    expected = (
        "Please improve session persistence so old conversations can" if fail_title else "Improve session persistence"
    )
    client = RecordingClient()
    async with (
        asyncio.timeout(30),
        acp.spawn_agent_process(cast(Client, client), sys.executable, *args) as (connection, _process),
    ):
        await connection.initialize(protocol_version=acp.PROTOCOL_VERSION)
        session = await connection.new_session(cwd=str(tmp_path), mcp_servers=[])
        for command in ("/help", "/checklist", "/title"):
            response = await connection.prompt(session_id=session.session_id, prompt=[acp.text_block(command)])
            assert response.stop_reason == "end_turn"
        assert not any(isinstance(update, schema.SessionInfoUpdate) for _, update in client.updates)

        response = await connection.prompt(session_id=session.session_id, prompt=[acp.text_block(first_prompt)])
        assert response.stop_reason == "end_turn"
        titles = [update.title for _, update in client.updates if isinstance(update, schema.SessionInfoUpdate)]
        assert titles == [expected]
        response = await connection.prompt(
            session_id=session.session_id, prompt=[acp.text_block("Now investigate a completely different task")]
        )
        assert response.stop_reason == "end_turn"
        titles = [update.title for _, update in client.updates if isinstance(update, schema.SessionInfoUpdate)]
        assert titles == [expected]

    resumed = RecordingClient()
    async with (
        asyncio.timeout(30),
        acp.spawn_agent_process(cast(Client, resumed), sys.executable, *args) as (connection, _process),
    ):
        await connection.initialize(protocol_version=acp.PROTOCOL_VERSION)
        await connection.load_session(session_id=session.session_id, cwd=str(tmp_path), mcp_servers=[])
        start = len(resumed.updates)
        response = await connection.prompt(session_id=session.session_id, prompt=[acp.text_block("/title")])
        assert response.stop_reason == "end_turn"
        text = "".join(
            update.content.text
            for _, update in resumed.updates[start:]
            if isinstance(update, schema.AgentMessageChunk) and isinstance(update.content, schema.TextContentBlock)
        )
        # The SDK may still be dispatching transcript replay when load_session returns.
        assert text.endswith(expected)
        titles = [update.title for _, update in resumed.updates if isinstance(update, schema.SessionInfoUpdate)]
        assert titles == [expected]
