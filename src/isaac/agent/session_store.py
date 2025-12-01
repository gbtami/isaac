"""Session persistence utilities for ACP session/load support.

Implements storing session metadata and session/update history to disk so that
`session/load` can replay prior turns, as described in the ACP Session Setup spec.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List

from acp.schema import SessionNotification


@dataclass
class SessionStore:
    """Persist session metadata and history for replay after restarts."""

    root: Path
    max_sessions: int = 50

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.cleanup()

    def session_dir(self, session_id: str) -> Path:
        path = self.root / session_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def persist_meta(
        self,
        session_id: str,
        cwd: Path,
        mcp_servers: Iterable[Any],
        *,
        current_mode: str,
    ) -> None:
        meta = {
            "cwd": str(cwd),
            "mcpServers": [self._dump_model(server) for server in (mcp_servers or [])],
            "mode": current_mode,
        }
        meta_path = self.session_dir(session_id) / "meta.json"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def load_meta(self, session_id: str) -> dict[str, Any]:
        meta_path = self.session_dir(session_id) / "meta.json"
        if not meta_path.exists():
            return {}
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def persist_update(self, session_id: str, note: SessionNotification) -> None:
        history_path = self.session_dir(session_id) / "history.jsonl"
        try:
            payload = note.model_dump(mode="json")
        except Exception:
            return
        with history_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")

    def load_history(self, session_id: str) -> List[SessionNotification]:
        history_path = self.session_dir(session_id) / "history.jsonl"
        if not history_path.exists():
            return []
        notes: list[SessionNotification] = []
        try:
            for line in history_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                payload = json.loads(line)
                notes.append(SessionNotification(**payload))
        except Exception:
            return notes
        return notes

    def cleanup(self) -> None:
        """Bound session storage by keeping only the newest `max_sessions` sessions."""
        try:
            entries = [
                (p, p.stat().st_mtime)
                for p in self.root.iterdir()
                if p.is_dir() and (p / "history.jsonl").exists()
            ]
        except FileNotFoundError:
            return

        if len(entries) <= self.max_sessions:
            return

        entries.sort(key=lambda t: t[1], reverse=True)
        for path, _ in entries[self.max_sessions :]:
            shutil.rmtree(path, ignore_errors=True)

    @staticmethod
    def _dump_model(obj: Any) -> dict[str, Any]:
        if hasattr(obj, "model_dump"):
            try:
                return obj.model_dump(mode="json")
            except Exception:
                pass
        try:
            return json.loads(json.dumps(obj, default=lambda o: getattr(o, "__dict__", {})))
        except Exception:
            return {}
