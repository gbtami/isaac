"""Exercise Isaac through the same stdio ACP transport used by Toad."""

import asyncio
import json
import sys
from pathlib import Path
from typing import cast

import acp
import pytest
from acp import Client, schema


class RecordingClient:
    """A minimal SDK client for the capabilities exercised here."""

    def __init__(self) -> None:
        self.updates: list[tuple[str, object]] = []
        self.approvals: list[schema.ToolCallUpdate] = []
        self.allow = True

    def on_connect(self, _connection: object) -> None:
        pass

    async def session_update(self, session_id: str, update: object, **_kwargs: object) -> None:
        self.updates.append((session_id, update))

    async def request_permission(
        self,
        session_id: str,
        tool_call: schema.ToolCallUpdate,
        options: list[schema.PermissionOption],
        **_kwargs: object,
    ) -> schema.RequestPermissionResponse:
        self.approvals.append(tool_call)
        kind = "allow_once" if self.allow else "reject_once"
        option = next(option for option in options if option.kind == kind)
        return schema.RequestPermissionResponse(
            outcome=schema.AllowedOutcome(outcome="selected", option_id=option.option_id)
        )


async def test_cli_initialization_and_model_selection(tmp_path: Path) -> None:
    """The installed entrypoint starts without provider credentials or a prompt."""
    client = RecordingClient()
    models = ("openai-codex:gpt-5.4", "anthropic:claude-sonnet-4-6")
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
        assert {command.name for command in commands[-1].available_commands} == {"help", "plan", "title", "compact"}
        assert session.config_options is not None
        assert session.config_options[0].current_value == models[0]
        selected = await connection.set_config_option(session_id=session.session_id, config_id="model", value=models[1])
        assert selected.config_options[0].current_value == models[1]


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
        acp.spawn_agent_process(cast(Client, client), sys.executable, str(fixture)) as (connection, _process),
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


@pytest.mark.parametrize("code_mode", [False, True])
async def test_plan_sidebar_and_commands(tmp_path: Path, code_mode: bool) -> None:
    """Native plan snapshots follow whole-plan edits across turns, including CodeMode."""
    client = RecordingClient()
    fixture = Path(__file__).with_name("acp_fixture.py")
    args = [str(fixture), *(["--code-mode"] if code_mode else [])]
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
        assert "No plan yet" in await prompt("/plan")
        assert "Usage: /plan" in await prompt("/plan unexpected-argument")
        assert "ready" == await prompt("/unknown-command")
        assert "ready" == await prompt("/help\nPlease explain this text.")
        items = [
            {"id": "inspect", "content": "Inspect code", "status": "in_progress"},
            {"id": "fix", "content": "Fix code", "status": "pending"},
        ]
        await prompt(json.dumps({"tool": "write_plan", "args": {"items": items}}))
        assert [entry.content for entry in plans()[-1].entries] == ["Inspect code", "Fix code"]
        assert [entry.status for entry in plans()[-1].entries] == ["in_progress", "pending"]
        assert "Inspect code" in await prompt("/plan")
        assert "No plan yet" in await prompt("/plan", session_id=other.session_id)

        await prompt(json.dumps({"tool": "write_plan", "args": {"items": list(reversed(items))}}))
        assert [entry.content for entry in plans()[-1].entries] == ["Fix code", "Inspect code"]
        items[0]["status"] = "completed"
        items[1]["status"] = "cancelled"
        await prompt(json.dumps({"tool": "write_plan", "args": {"items": items}}))
        assert [(entry.content, entry.status) for entry in plans()[-1].entries] == [("Inspect code", "completed")]
        await prompt(json.dumps({"tool": "write_plan", "args": {"items": []}}))
        assert plans()[-1].entries == []


@pytest.mark.parametrize("fail_summary", [False, True])
async def test_compact_commits_history_for_next_turn(tmp_path: Path, fail_summary: bool) -> None:
    """The command uses Harness compaction and the adapter commits the resulting history."""
    client = RecordingClient()
    fixture = Path(__file__).with_name("acp_fixture.py")
    args = [str(fixture), *(["--fail-summary"] if fail_summary else [])]
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
        assert "Compacted" in await prompt("/compact preserve the current task")
        after = int(await prompt(inspect))
        assert after < before


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
        assert any(isinstance(update, schema.UserMessageChunk) for _, update in resumed.updates)
        assert any(isinstance(update, schema.AgentMessageChunk) for _, update in resumed.updates)
        assert any(isinstance(update, schema.ToolCallStart) for _, update in resumed.updates)
        plans = [update for _, update in resumed.updates if isinstance(update, schema.AgentPlanUpdate)]
        assert [(entry.content, entry.status) for entry in plans[-1].entries] == [("Continue the task", "in_progress")]

        start = len(resumed.updates)
        response = await connection.prompt(session_id=session.session_id, prompt=[acp.text_block("/plan")])
        assert response.stop_reason == "end_turn"
        assert any(
            isinstance(update, schema.AgentMessageChunk)
            and isinstance(update.content, schema.TextContentBlock)
            and "Continue the task" in update.content.text
            for _, update in resumed.updates[start:]
        )
        start = len(resumed.updates)
        response = await connection.prompt(
            session_id=session.session_id,
            prompt=[acp.text_block(json.dumps({"tool": "history_length", "args": {}}))],
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
        # The SDK dispatches session notifications independently of the request response.
        await asyncio.sleep(0)
        plans = [
            update
            for received, update in resumed.updates
            if received == untouched.session_id and isinstance(update, schema.AgentPlanUpdate)
        ]
        assert plans[-1].entries == []


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
        for command in ("/help", "/plan", "/title"):
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
