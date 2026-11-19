"""Durable storage implementing Harness's public ACP SessionStore protocol."""

import asyncio
import logging
import os
import sqlite3
from contextlib import closing
from dataclasses import replace
from pathlib import Path

from acp import text_block, update_agent_message
from pydantic import TypeAdapter
from pydantic_ai.messages import ModelResponse, TextPart, repair_messages
from pydantic_ai_harness.experimental.acp import StoredSession
from pydantic_ai_harness.step_persistence import SqliteStepStore, StepPersistence

from isaac.config import SessionSettings

SESSION = TypeAdapter(StoredSession)
logger = logging.getLogger(__name__)


class SqliteSessionStore:
    """Save complete snapshots atomically; leave ACP history/replay to Harness."""

    def __init__(self, database: Path) -> None:
        self.database = database
        database.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        database.touch(mode=0o600, exist_ok=True)
        database.chmod(0o600)
        with closing(sqlite3.connect(database)) as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, snapshot BLOB NOT NULL)")
            connection.execute("CREATE TABLE IF NOT EXISTS session_titles (id TEXT PRIMARY KEY, title TEXT NOT NULL)")
            connection.commit()
        self.steps = SqliteStepStore(database=database, max_snapshots_per_run=2)

    def checkpoints(self, session_id: str) -> StepPersistence[SessionSettings]:
        return StepPersistence(
            store=self.steps,
            capture_frontier=True,
            metadata={"acp_session_id": session_id, "pid": str(os.getpid())},
        )

    async def save(self, session_id: str, session: StoredSession) -> None:
        await asyncio.to_thread(self._save, session_id, session)

    def _save(self, session_id: str, session: StoredSession) -> None:
        snapshot = SESSION.dump_json(session)
        with closing(sqlite3.connect(self.database)) as connection:
            connection.execute(
                "INSERT INTO sessions (id, snapshot) VALUES (?, ?) "
                "ON CONFLICT(id) DO UPDATE SET snapshot = excluded.snapshot",
                (session_id, snapshot),
            )
            connection.commit()

    async def load(self, session_id: str) -> StoredSession | None:
        session = await asyncio.to_thread(self._load, session_id)
        if session is None:
            return None
        try:
            return await self._recover(session_id, session)
        except (sqlite3.Error, ValueError):
            logger.warning("Could not read Isaac checkpoints for session %s", session_id, exc_info=True)
            return session

    async def _recover(self, session_id: str, session: StoredSession) -> StoredSession:
        # Sessions created before checkpointing already have a core conversation ID.
        conversation_id = next(
            (message.conversation_id for message in reversed(session.messages) if message.conversation_id), session_id
        )
        committed = {message.run_id for message in session.messages}
        for run in reversed(await self.steps.list_runs(conversation_id=conversation_id)):
            if run.parent_run_id or run.metadata.get("acp_session_id") != session_id:
                continue
            if run.run_id in committed or _process_is_running(int(run.metadata.get("pid", "0"))):
                return session
            checkpoint = await self.steps.latest_snapshot(run_id=run.run_id, include_interrupted=True)
            if checkpoint is not None and checkpoint.messages:
                break
        else:
            return session

        messages = list(checkpoint.messages)
        if isinstance(tail := messages[-1], ModelResponse) and tail.state == "suspended":
            messages[-1] = replace(tail, state="interrupted")
        messages = repair_messages(messages)
        pending = await self.steps.list_unresolved_tool_effects(run_id=run.run_id)
        notice = "Recovered a checkpoint from an interrupted turn. Ask Isaac to continue when ready."
        if pending:
            tools = ", ".join(sorted({effect.tool_name for effect in pending}))
            notice += f" These tools were still running and may have had side effects: {tools}."
        notice += " Check for partial changes before retrying edits or commands."
        messages.append(
            ModelResponse(parts=[TextPart(notice)], run_id=run.run_id, conversation_id=checkpoint.conversation_id)
        )
        return replace(session, messages=messages, updates=[*session.updates, update_agent_message(text_block(notice))])

    def _load(self, session_id: str) -> StoredSession | None:
        with closing(sqlite3.connect(self.database)) as connection:
            row = connection.execute("SELECT snapshot FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return SESSION.validate_json(row[0]) if row is not None else None

    async def get_title(self, session_id: str) -> str | None:
        return await asyncio.to_thread(self._get_title, session_id)

    def _get_title(self, session_id: str) -> str | None:
        with closing(sqlite3.connect(self.database)) as connection:
            row = connection.execute("SELECT title FROM session_titles WHERE id = ?", (session_id,)).fetchone()
        return str(row[0]) if row is not None else None

    async def save_title(self, session_id: str, title: str) -> str:
        """Keep the first title, including when two clients name the same session."""
        return await asyncio.to_thread(self._save_title, session_id, title)

    def _save_title(self, session_id: str, title: str) -> str:
        with closing(sqlite3.connect(self.database)) as connection:
            connection.execute("INSERT OR IGNORE INTO session_titles (id, title) VALUES (?, ?)", (session_id, title))
            connection.commit()
            row = connection.execute("SELECT title FROM session_titles WHERE id = ?", (session_id,)).fetchone()
        assert row is not None
        return str(row[0])


def _process_is_running(pid: int) -> bool:
    if pid == os.getpid():
        return True
    if pid <= 0 or os.name != "posix":
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True
