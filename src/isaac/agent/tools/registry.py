"""Tool registry and metadata for ACP tool exposure."""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, Type

from .apply_patch import apply_patch
from .code_search import code_search
from .edit_file import edit_file
from .file_summary import file_summary
from .list_files import list_files
from .read_file import read_file
from .run_command import run_command
from .fetch_url import DEFAULT_FETCH_MAX_BYTES, DEFAULT_FETCH_TIMEOUT, fetch_url
from isaac.agent.subagents import (
    DELEGATE_TOOL_ARG_MODELS,
    DELEGATE_TOOL_DESCRIPTIONS,
    DELEGATE_TOOL_HANDLERS,
)
from .args import (
    ApplyPatchArgs,
    CodeSearchArgs,
    EditFileArgs,
    FetchUrlArgs,
    FileSummaryArgs,
    ListFilesArgs,
    ReadFileArgs,
    RunCommandArgs,
)

ToolHandler = Callable[..., Awaitable[dict]]

DEFAULT_TOOL_TIMEOUT_S = 10.0
RUN_COMMAND_TIMEOUT_S = 60.0

# Tools permitted for planning delegate (read-only, non-destructive)
READ_ONLY_TOOLS = {
    "list_files",
    "read_file",
    "file_summary",
    "code_search",
    "fetch_url",
}

TOOL_HANDLERS: Dict[str, ToolHandler] = {
    "list_files": list_files,
    "read_file": read_file,
    "run_command": run_command,
    "edit_file": edit_file,
    "code_search": code_search,
    "apply_patch": apply_patch,
    "file_summary": file_summary,
    "fetch_url": fetch_url,
    **DELEGATE_TOOL_HANDLERS,
}

# Pydantic argument models for each tool, used for schema generation and validation.
TOOL_ARG_MODELS: Dict[str, Type[Any]] = {
    "list_files": ListFilesArgs,
    "read_file": ReadFileArgs,
    "run_command": RunCommandArgs,
    "edit_file": EditFileArgs,
    "apply_patch": ApplyPatchArgs,
    "file_summary": FileSummaryArgs,
    "code_search": CodeSearchArgs,
    "fetch_url": FetchUrlArgs,
    **DELEGATE_TOOL_ARG_MODELS,
}

TOOL_DESCRIPTIONS: Dict[str, str] = {
    "list_files": "List files and directories (defaults to current directory; set directory if you need a different root).",
    "read_file": "Read a file with optional line range",
    "run_command": "Execute a shell command and return its output (include the full command string).",
    "edit_file": "Replace the contents of a file (requires a path and content; path must not be a directory).",
    "apply_patch": "Apply a unified diff patch to a file",
    "file_summary": "Summarize a file (head/tail and line count)",
    "code_search": "Search for a pattern in files using ripgrep",
    "fetch_url": "Fetch a URL (https only) with size/time limits; useful for docs and API refs.",
    **DELEGATE_TOOL_DESCRIPTIONS,
}

TOOL_REQUIRED_ARGS: Dict[str, list[str]] = {
    name: list(model.model_json_schema().get("required", []))  # type: ignore[attr-defined]
    for name, model in TOOL_ARG_MODELS.items()
}

_FULL_TOOL_ORDER = tuple(TOOL_HANDLERS.keys())
_READ_ONLY_TOOL_ORDER = tuple(name for name in _FULL_TOOL_ORDER if name in READ_ONLY_TOOLS)

__all__ = [
    "DEFAULT_FETCH_MAX_BYTES",
    "DEFAULT_FETCH_TIMEOUT",
    "DEFAULT_TOOL_TIMEOUT_S",
    "RUN_COMMAND_TIMEOUT_S",
    "READ_ONLY_TOOLS",
    "TOOL_HANDLERS",
    "TOOL_ARG_MODELS",
    "TOOL_DESCRIPTIONS",
    "TOOL_REQUIRED_ARGS",
    "_FULL_TOOL_ORDER",
    "_READ_ONLY_TOOL_ORDER",
    "ToolHandler",
]
