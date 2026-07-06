"""Pydantic argument models for Isaac tool execution.

Pydantic AI derives provider-facing schemas from the wrapper signatures in
``registration.py``. These models validate direct ACP tool-call blocks and keep
``run_tool`` from calling handlers with malformed arguments.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
from .fetch_url import DEFAULT_FETCH_MAX_BYTES, DEFAULT_FETCH_TIMEOUT


class ListFilesArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")

    directory: str = Field(
        ".",
        description="Directory to list, default '.'",
    )
    recursive: bool = Field(
        True,
        description="Whether to list recursively",
    )


class ReadFileArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")

    path: str = Field(
        ...,
        min_length=1,
        description="Path to the file to read",
    )
    start: int | None = Field(
        None,
        description="Starting line number (1-based)",
        ge=1,
    )
    lines: int | None = Field(
        None,
        description="Number of lines to read from start",
        ge=1,
    )
    max_lines: int | None = Field(
        None,
        description=(
            "Maximum lines to return when reading from start or the beginning. "
            "Defaults to a safe bounded value; use next_start from truncated results to continue."
        ),
        ge=1,
    )


class RunCommandArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")

    command: str = Field(
        ...,
        min_length=1,
        description="Command to run (do not leave blank).",
    )
    cwd: str | None = Field(
        None,
        description="Working directory (optional; defaults to session cwd)",
    )
    timeout: float | None = Field(
        None,
        description="Timeout in seconds",
        gt=0,
    )


class EditFileArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")

    path: str = Field(
        ...,
        min_length=1,
        description="Path to the file to edit (relative paths are allowed).",
    )
    content: str = Field(
        ...,
        min_length=1,
        description=(
            "Full file contents to write (cannot be empty). "
            "Do not use edit_file to create directories; use run_command mkdir -p instead."
        ),
    )
    create: bool = Field(
        True,
        description="Create the file if it does not exist (default: true).",
    )
    expected_sha256: str | None = Field(
        None,
        description="Optional SHA-256 hash that the existing file must match before writing.",
    )


class ApplyPatchArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")

    path: str = Field(
        ...,
        min_length=1,
        description="Path to the file to patch",
    )
    patch: str = Field(
        ...,
        min_length=1,
        description="Unified diff patch text",
    )
    strip: int | None = Field(
        None,
        description="Strip leading path components (patch -p)",
        ge=0,
    )
    expected_sha256: str | None = Field(
        None,
        description="Optional SHA-256 hash that the existing file must match before patching.",
    )


class FileSummaryArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")

    path: str = Field(
        ...,
        min_length=1,
        description="Path to summarize",
    )
    head_lines: int | None = Field(
        20,
        description="Number of head lines to include",
        ge=0,
    )
    tail_lines: int | None = Field(
        20,
        description="Number of tail lines to include",
        ge=0,
    )


class CodeSearchArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")

    pattern: str = Field(
        ...,
        min_length=1,
        description="Pattern to search for",
    )
    directory: str = Field(
        ".",
        description="Root directory to search",
    )
    glob: str | None = Field(
        None,
        description="Glob pattern filter",
    )
    case_sensitive: bool = Field(
        True,
        description="Case sensitive search",
    )
    timeout: float | None = Field(
        None,
        description="Timeout in seconds",
        gt=0,
    )
    max_results: int | None = Field(
        None,
        description="Maximum number of search matches to return; defaults to a bounded result set.",
        ge=1,
    )


class FetchUrlArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")

    url: str = Field(
        ...,
        min_length=1,
        description="HTTP or HTTPS URL to fetch",
    )
    max_bytes: int = Field(
        DEFAULT_FETCH_MAX_BYTES,
        description="Maximum bytes to read from the response body",
        ge=1,
    )
    timeout: float | None = Field(
        DEFAULT_FETCH_TIMEOUT,
        description="Request timeout in seconds",
        gt=0,
    )
