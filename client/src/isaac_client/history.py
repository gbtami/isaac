"""Client-local names and IDs for sessions that can be reopened over ACP."""

import hashlib
import json
import os
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from acp.client.connection import ClientSideConnection
from platformdirs import user_state_path


@dataclass(frozen=True)
class SessionEntry:
    session_id: str
    title: str | None
    updated_at: datetime | None

    def label(self) -> str:
        label = f"{self.title or 'Untitled session'}  [{self.session_id}]"
        if self.updated_at:
            label += f"  {self.updated_at.astimezone():%Y-%m-%d %H:%M}"
        return label


class SessionHistory:
    def __init__(self, command: Sequence[str], cwd: Path) -> None:
        override = os.environ.get("ISAAC_CLIENT_STATE_DIR")
        state = Path(override).expanduser() if override else user_state_path("isaac-client", appauthor=False)
        state.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.database = state / "sessions.sqlite3"
        self.database.touch(mode=0o600, exist_ok=True)
        self.agent = hashlib.sha256(json.dumps(list(command)).encode()).hexdigest()
        self.cwd = str(cwd)
        with closing(sqlite3.connect(self.database)) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS sessions ("
                "agent TEXT, cwd TEXT, session_id TEXT, title TEXT, updated_at TEXT, "
                "PRIMARY KEY (agent, cwd, session_id))"
            )

    def recent(self) -> list[SessionEntry]:
        with closing(sqlite3.connect(self.database)) as db:
            rows = db.execute(
                "SELECT session_id, title, updated_at FROM sessions WHERE agent = ? AND cwd = ? ORDER BY updated_at DESC",
                (self.agent, self.cwd),
            )
            return [
                SessionEntry(session_id, title, datetime.fromisoformat(updated)) for session_id, title, updated in rows
            ]

    async def discover(self, connection: ClientSideConnection) -> list[SessionEntry]:
        entries = {entry.session_id: entry for entry in self.recent()}
        cursor = None
        seen: set[str] = set()
        while True:
            response = await connection.list_sessions(cwd=self.cwd, cursor=cursor)
            for item in response.sessions:
                if Path(item.cwd) != Path(self.cwd):
                    continue
                local = entries.get(item.session_id)
                updated = local.updated_at if local else None
                if item.updated_at:
                    try:
                        reported = datetime.fromisoformat(item.updated_at)
                    except ValueError:
                        pass
                    else:
                        reported = reported.astimezone(UTC)
                        updated = max(updated, reported) if updated else reported
                entries[item.session_id] = SessionEntry(
                    item.session_id, item.title or (local.title if local else None), updated
                )
            cursor = response.next_cursor
            if cursor is None or cursor in seen:
                break
            seen.add(cursor)
        return sorted(
            entries.values(), key=lambda entry: entry.updated_at or datetime.min.replace(tzinfo=UTC), reverse=True
        )

    def remember(self, session_id: str, title: str | None = None) -> None:
        with closing(sqlite3.connect(self.database)) as db, db:
            db.execute(
                "INSERT INTO sessions (agent, cwd, session_id, title, updated_at) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT (agent, cwd, session_id) DO UPDATE SET "
                "title = COALESCE(excluded.title, sessions.title), updated_at = excluded.updated_at",
                (self.agent, self.cwd, session_id, title, datetime.now(UTC).isoformat()),
            )
