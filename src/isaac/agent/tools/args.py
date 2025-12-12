"""Pydantic argument models for ACP/pydantic-ai tools.

These models serve two purposes:
- Provide strong JSON schemas to pydantic-ai for tool call validation/retries.
- Drive ACP tool capability descriptions without duplicating schema definitions.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


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
    )
    lines: int | None = Field(
        None,
        description="Number of lines to read",
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
        description="New content to write into the file.",
    )
    create: bool = Field(
        True,
        description="Create the file if it does not exist (default: true).",
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
    )
    tail_lines: int | None = Field(
        20,
        description="Number of tail lines to include",
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
    )
