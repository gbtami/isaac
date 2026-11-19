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

from platformdirs import user_state_path


@dataclass(frozen=True)
class SessionEntry:
    session_id: str
    title: str | None


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
                "SELECT session_id, title FROM sessions WHERE agent = ? AND cwd = ? ORDER BY updated_at DESC",
                (self.agent, self.cwd),
            )
            return [SessionEntry(session_id, title) for session_id, title in rows]

    def remember(self, session_id: str, title: str | None = None) -> None:
        with closing(sqlite3.connect(self.database)) as db, db:
            db.execute(
                "INSERT INTO sessions (agent, cwd, session_id, title, updated_at) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT (agent, cwd, session_id) DO UPDATE SET "
                "title = COALESCE(excluded.title, sessions.title), updated_at = excluded.updated_at",
                (self.agent, self.cwd, session_id, title, datetime.now(UTC).isoformat()),
            )
