"""Durable storage implementing Harness's public ACP SessionStore protocol."""

import asyncio
import sqlite3
from contextlib import closing
from pathlib import Path

from pydantic import TypeAdapter
from pydantic_ai_harness.experimental.acp import StoredSession

SESSION = TypeAdapter(StoredSession)


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
        return await asyncio.to_thread(self._load, session_id)

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
