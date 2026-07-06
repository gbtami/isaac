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

from isaac.agent.tools.safety import (
    BinaryFileError,
    PathAccessError,
    ProtectedPathError,
    ensure_text_file,
    ensure_text_target,
    resolve_workspace_path,
)


def resolve_path_for_session(session_cwds: Dict[str, Path], session_id: str, path_str: str) -> Path:
    """Resolve a path under the session cwd with symlink-aware containment."""

    base = session_cwds.get(session_id, Path.cwd())
    return resolve_workspace_path(base, path_str)


async def read_text_file(session_cwds: Dict[str, Path], params: ReadTextFileRequest) -> ReadTextFileResponse:
    """Implement fs/read_text_file with absolute path resolution and 1-based lines."""

    try:
        path = resolve_path_for_session(session_cwds, params.session_id, params.path)
        if not path.exists() or not path.is_file():
            return ReadTextFileResponse(content="")
        ensure_text_file(path)
        lines = path.read_text(encoding="utf-8").splitlines()
        start = max((params.line or 1) - 1, 0)
        limit = params.limit or len(lines)
        selected = lines[start : start + limit]
        return ReadTextFileResponse(content="\n".join(selected))
    except (BinaryFileError, PathAccessError, OSError, UnicodeDecodeError):
        return ReadTextFileResponse(content="")


async def write_text_file(session_cwds: Dict[str, Path], params: WriteTextFileRequest) -> WriteTextFileResponse:
    """Implement fs/write_text_file honoring the session cwd."""

    try:
        path = resolve_path_for_session(session_cwds, params.session_id, params.path)
        ensure_text_target(path, session_cwds.get(params.session_id, Path.cwd()))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(params.content, encoding="utf-8")
        return WriteTextFileResponse()
    except (BinaryFileError, PathAccessError, ProtectedPathError, OSError, UnicodeDecodeError):
        return WriteTextFileResponse()
