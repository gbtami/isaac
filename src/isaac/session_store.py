"""Durable storage implementing Harness's public ACP SessionStore protocol."""

import asyncio
import logging
import os
import sqlite3
from collections.abc import Awaitable, Callable
from contextlib import closing
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path

from acp import Client, RequestError, text_block, update_agent_message
from pydantic import TypeAdapter
from pydantic_ai import RunContext
from pydantic_ai.messages import ModelResponse, TextPart, repair_messages
from pydantic_ai_harness.experimental.acp import ACPSession, StoredSession
from pydantic_ai_harness.step_persistence import SqliteStepStore, StepPersistence

from isaac.config import SessionSettings

SESSION = TypeAdapter(StoredSession)
logger = logging.getLogger(__name__)
_CHECKPOINT_REVISION = "isaac_session_revision"
_REOPEN = "Another Isaac process updated this session. Reopen the session to continue from the latest saved state."
_PLANNING = "EXISTS (SELECT 1 FROM planning_sessions WHERE planning_sessions.id = sessions.id)"


class _SaveConflict(RuntimeError):
    """The stored snapshot advanced while this process held an earlier revision."""


@dataclass
class _OpenedSession:
    revision: int | None
    planning: bool = False
    client: Client | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass
class _SessionCheckpoints(StepPersistence[SessionSettings]):
    check_revision: Callable[[], Awaitable[int | None]] = field(kw_only=True)

    async def before_run(self, ctx: RunContext[SessionSettings]) -> None:
        self.metadata = {**self.metadata, _CHECKPOINT_REVISION: str(await self.check_revision())}
        await super().before_run(ctx)


class SqliteSessionStore:
    """Save complete snapshots atomically; leave ACP history/replay to Harness."""

    def __init__(self, database: Path) -> None:
        self.database = database
        self._opened: dict[str, _OpenedSession] = {}
        self._loaded_state: ContextVar[tuple[str, int, bool] | None] = ContextVar("loaded_state", default=None)
        database.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        database.touch(mode=0o600, exist_ok=True)
        database.chmod(0o600)
        with closing(sqlite3.connect(database)) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS sessions ("
                "id TEXT PRIMARY KEY, snapshot BLOB NOT NULL, revision INTEGER NOT NULL)"
            )
            connection.execute("CREATE TABLE IF NOT EXISTS session_titles (id TEXT PRIMARY KEY, title TEXT NOT NULL)")
            connection.execute("CREATE TABLE IF NOT EXISTS planning_sessions (id TEXT PRIMARY KEY)")
            connection.commit()
        self.steps = SqliteStepStore(database=database, max_snapshots_per_run=2)

    def _open_for_save(self, session_id: str) -> _OpenedSession:
        loaded = self._loaded_state.get()
        if loaded is not None and loaded[0] == session_id:
            self._loaded_state.set(None)
            self._opened[session_id] = _OpenedSession(loaded[1], loaded[2])
        elif session_id not in self._opened:
            self._opened[session_id] = _OpenedSession(None)
        return self._opened[session_id]

    def checkpoints(self, session: ACPSession) -> StepPersistence[SessionSettings]:
        session_id = session.session_id
        opened = self._open_for_save(session_id)
        opened.client = session.client
        return _SessionCheckpoints(
            store=self.steps,
            capture_frontier=True,
            metadata={"acp_session_id": session_id, "pid": str(os.getpid())},
            check_revision=partial(self._check_revision, session_id, opened),
        )

    async def _check_revision(self, session_id: str, opened: _OpenedSession) -> int | None:
        async with opened.lock:
            if await asyncio.to_thread(self._version, session_id) != (opened.revision, opened.planning):
                raise RequestError(-32603, _REOPEN)
            return opened.revision

    def _version(self, session_id: str) -> tuple[int | None, bool]:
        with closing(sqlite3.connect(self.database)) as connection:
            return self._read_version(connection, session_id)

    @staticmethod
    def _read_version(connection: sqlite3.Connection, session_id: str) -> tuple[int | None, bool]:
        row = connection.execute(f"SELECT revision, {_PLANNING} FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return (int(row[0]), bool(row[1])) if row is not None else (None, False)

    async def save(self, session_id: str, session: StoredSession) -> None:
        opened = self._open_for_save(session_id)
        try:
            await self._write(opened, partial(self._save, session_id, session, opened))
        except _SaveConflict:
            if opened.client is not None:
                try:
                    await opened.client.session_update(
                        session_id=session_id,
                        update=update_agent_message(
                            text_block(
                                f"\n\nIsaac skipped this save. {_REOPEN} Review any tool changes before retrying."
                            )
                        ),
                    )
                except Exception:
                    logger.warning("Could not report a session save conflict to the client", exc_info=True)
            raise

    async def _write(self, opened: _OpenedSession, operation: Callable[[], None]) -> None:
        async with opened.lock:
            write = asyncio.create_task(asyncio.to_thread(operation))
            try:
                await asyncio.shield(write)
            except asyncio.CancelledError:
                # The worker may still commit. Keep the local revision and write order in sync.
                await asyncio.gather(write, return_exceptions=True)
                raise

    def _save(self, session_id: str, session: StoredSession, opened: _OpenedSession) -> None:
        snapshot = SESSION.dump_json(session)
        revision = opened.revision
        with closing(sqlite3.connect(self.database)) as connection, connection:
            if revision is None:
                changed = connection.execute(
                    "INSERT INTO sessions (id, snapshot, revision) VALUES (?, ?, 0) ON CONFLICT(id) DO NOTHING",
                    (session_id, snapshot),
                )
            else:
                changed = connection.execute(
                    "UPDATE sessions SET snapshot = ?, revision = revision + 1 "
                    f"WHERE id = ? AND revision = ? AND {_PLANNING} = ?",
                    (snapshot, session_id, revision, opened.planning),
                )
            if changed.rowcount != 1:
                raise _SaveConflict(_REOPEN)
        opened.revision = 0 if revision is None else revision + 1

    async def load(self, session_id: str) -> StoredSession | None:
        record = await asyncio.to_thread(self._load, session_id)
        if record is None:
            return None
        session, revision, planning = record
        # Harness reads the snapshot before draining an old live turn. Hand its revision and mode
        # to the session factory without changing the old turn's expected state.
        self._loaded_state.set((session_id, revision, planning))
        try:
            return await self._recover(session_id, session, revision)
        except (sqlite3.Error, ValueError):
            logger.warning("Could not read Isaac checkpoints for session %s", session_id, exc_info=True)
            return session

    async def _recover(self, session_id: str, session: StoredSession, revision: int) -> StoredSession:
        committed = {message.run_id for message in session.messages}
        for run in reversed(await self.steps.list_runs(conversation_id=session_id)):
            if run.parent_run_id or run.metadata.get("acp_session_id") != session_id:
                continue
            if run.run_id in committed:
                return session
            if run.metadata.get(_CHECKPOINT_REVISION) != str(revision):
                continue
            if _process_is_running(int(run.metadata.get("pid", "0"))):
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

    async def read_snapshot(self, session_id: str) -> StoredSession | None:
        """Read a committed snapshot for metadata without changing the revision used for saves."""
        record = await asyncio.to_thread(self._load, session_id)
        return record[0] if record is not None else None

    def _load(self, session_id: str) -> tuple[StoredSession, int, bool] | None:
        with closing(sqlite3.connect(self.database)) as connection:
            row = connection.execute(
                f"SELECT snapshot, revision, {_PLANNING} FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
        return (SESSION.validate_json(row[0]), int(row[1]), bool(row[2])) if row is not None else None

    def is_planning(self, session_id: str) -> bool:
        return self._opened[session_id].planning

    async def set_planning(self, session_id: str, planning: bool) -> None:
        opened = self._opened[session_id]
        try:
            await self._write(opened, partial(self._set_planning, session_id, planning, opened))
        except _SaveConflict as exc:
            raise RequestError(-32603, _REOPEN) from exc

    def _set_planning(self, session_id: str, planning: bool, opened: _OpenedSession) -> None:
        # Keep the checkpoint baseline until the adapter saves a new conversation snapshot.
        with closing(sqlite3.connect(self.database)) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            if opened.revision is None or self._read_version(connection, session_id) != (
                opened.revision,
                opened.planning,
            ):
                raise _SaveConflict(_REOPEN)
            if planning:
                connection.execute("INSERT INTO planning_sessions (id) VALUES (?)", (session_id,))
            else:
                connection.execute("DELETE FROM planning_sessions WHERE id = ?", (session_id,))
        opened.planning = planning

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
