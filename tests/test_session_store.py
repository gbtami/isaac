"""Persistence round-trips typed ACP/Pydantic AI data without silently replacing corruption."""

import os
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from acp import schema
from pydantic import ValidationError
from pydantic_ai.messages import BinaryImage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai_harness.experimental.acp import StoredSession
from pydantic_ai_harness.step_persistence import ContinuableSnapshot, RunRecord

from isaac.session_store import SqliteSessionStore


async def test_typed_snapshot_survives_reopening_and_update(tmp_path: Path) -> None:
    database = tmp_path / "state" / "sessions.sqlite3"
    store = SqliteSessionStore(database)
    snapshot = StoredSession(
        messages=[
            ModelRequest(
                parts=[UserPromptPart(content=["Inspect this", BinaryImage(data=b"\x00\xff", media_type="image/png")])]
            ),
            ModelResponse(parts=[TextPart("Saved response")]),
        ],
        updates=[
            schema.AgentMessageChunk(
                session_update="agent_message_chunk",
                content=schema.TextContentBlock(type="text", text="Saved response"),
            )
        ],
        model="openai:example",
    )
    await store.save("first", snapshot)
    reopened = SqliteSessionStore(database)
    assert await reopened.load("first") == snapshot
    assert await reopened.load("unknown") is None
    assert await reopened.get_title("first") is None
    assert await reopened.save_title("first", "Inspect image rendering") == "Inspect image rendering"
    assert await store.get_title("first") == "Inspect image rendering"
    assert await store.save_title("first", "A later title") == "Inspect image rendering"
    await reopened.save("second", StoredSession(model="anthropic:example"))
    await reopened.save("first", StoredSession(model="openai:updated"))
    assert await store.load("first") == StoredSession(model="openai:updated")
    assert await SqliteSessionStore(database).get_title("first") == "Inspect image rendering"
    assert await store.load("second") == StoredSession(model="anthropic:example")
    if os.name == "posix":
        assert database.stat().st_mode & 0o777 == 0o600


async def test_corrupt_snapshot_is_reported_without_overwriting_it(tmp_path: Path) -> None:
    database = tmp_path / "sessions.sqlite3"
    store = SqliteSessionStore(database)
    with closing(sqlite3.connect(database)) as connection:
        connection.execute("INSERT INTO sessions (id, snapshot) VALUES (?, ?)", ("broken", b"invalid JSON"))
        connection.commit()
    with pytest.raises(ValidationError):
        await store.load("broken")
    with closing(sqlite3.connect(database)) as connection:
        assert connection.execute("SELECT snapshot FROM sessions WHERE id = 'broken'").fetchone() == (b"invalid JSON",)


async def test_checkpoint_survives_a_second_interruption_before_the_first_step(tmp_path: Path) -> None:
    store = SqliteSessionStore(tmp_path / "sessions.sqlite3")
    await store.save("session", StoredSession())
    for run_id in ("with-checkpoint", "without-checkpoint"):
        await store.steps.register_run(
            RunRecord(run_id=run_id, conversation_id="session", metadata={"acp_session_id": "session"})
        )
    await store.steps.save_snapshot(
        ContinuableSnapshot(
            run_id="with-checkpoint",
            step_index=1,
            conversation_id="session",
            messages=[ModelRequest(parts=[UserPromptPart("Original prompt")], conversation_id="session")],
        )
    )

    recovered = await store.load("session")
    assert recovered is not None and len(recovered.messages) == 2
    assert recovered.messages[-1].run_id == "with-checkpoint"
