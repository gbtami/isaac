"""Filesystem helpers mapped to ACP file-system endpoints.

See: https://agentclientprotocol.com/protocol/file-system for required behavior.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from acp import (
    ReadTextFileRequest,
    ReadTextFileResponse,
    WriteTextFileRequest,
    WriteTextFileResponse,
)


def resolve_path_for_session(session_cwds: Dict[str, Path], session_id: str, path_str: str) -> Path:
    """Resolve an absolute path anchored to the session cwd (per File System spec)."""
    base = session_cwds.get(session_id, Path.cwd())
    path = Path(path_str)
    if not path.is_absolute():
        path = base / path
    return path.expanduser().resolve()


async def read_text_file(
    session_cwds: Dict[str, Path], params: ReadTextFileRequest
) -> ReadTextFileResponse:
    """Implement fs/read_text_file with absolute path resolution and 1-based lines."""
    path = resolve_path_for_session(session_cwds, params.sessionId, params.path)
    if not path.exists() or not path.is_file():
        return ReadTextFileResponse(content="")

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        start = max((params.line or 1) - 1, 0)
        limit = params.limit or len(lines)
        selected = lines[start : start + limit]
        return ReadTextFileResponse(content="\n".join(selected))
    except Exception:
        return ReadTextFileResponse(content="")


async def write_text_file(
    session_cwds: Dict[str, Path], params: WriteTextFileRequest
) -> WriteTextFileResponse:
    """Implement fs/write_text_file honoring the session cwd."""
    path = resolve_path_for_session(session_cwds, params.sessionId, params.path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(params.content, encoding="utf-8")
        return WriteTextFileResponse()
    except Exception:
        return WriteTextFileResponse()
