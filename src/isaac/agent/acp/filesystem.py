"""File system handlers for ACP."""

from __future__ import annotations

from typing import Any

from acp import ReadTextFileResponse, WriteTextFileResponse
from acp.schema import ReadTextFileRequest, WriteTextFileRequest

from isaac.agent.fs import read_text_file, write_text_file


class FileSystemMixin:
    async def read_text_file(self, path: str, session_id: str, **kwargs: Any) -> ReadTextFileResponse:
        """Serve fs/read_text_file to clients (File System section)."""
        params = ReadTextFileRequest(path=path, session_id=session_id, field_meta=kwargs or None)
        return await read_text_file(self._session_cwds, params)

    async def write_text_file(self, content: str, path: str, session_id: str, **kwargs: Any) -> WriteTextFileResponse:
        """Serve fs/write_text_file to clients (File System section)."""
        params = WriteTextFileRequest(content=content, path=path, session_id=session_id, field_meta=kwargs or None)
        return await write_text_file(self._session_cwds, params)
