"""Local tool registry and ACP tool descriptions."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List

from .apply_patch import apply_patch
from .code_search import code_search
from .edit_file import edit_file
from .file_summary import file_summary
from .list_files import list_files
from .read_file import read_file
from .run_command import run_command


@dataclass
class ToolParameter:
    type: str
    properties: dict[str, Any]
    required: list[str]


@dataclass
class Tool:
    function: str
    description: str
    parameters: ToolParameter


ToolHandler = Callable[..., Awaitable[dict]]

TOOL_HANDLERS: Dict[str, ToolHandler] = {
    "list_files": list_files,
    "read_file": read_file,
    "run_command": run_command,
    "edit_file": edit_file,
    "code_search": code_search,
    "apply_patch": apply_patch,
    "file_summary": file_summary,
}

# Tools permitted for planning delegate (read-only, non-destructive)
READ_ONLY_TOOLS = {
    "list_files",
    "read_file",
    "file_summary",
    "code_search",
}

TOOL_REQUIRED_ARGS: Dict[str, list[str]] = {
    "list_files": ["directory"],
    "read_file": ["file_path"],
    "run_command": ["command"],
    "edit_file": ["file_path", "new_content"],
    "apply_patch": ["file_path", "patch"],
    "code_search": ["pattern", "directory"],
    "file_summary": ["file_path"],
}


def get_tools() -> List[Any]:
    """Return ACP tool descriptions (with graceful fallback when schema lacks Tool)."""
    base_tools = [
        Tool(
            function="list_files",
            description="List files and directories recursively",
            parameters=ToolParameter(
                type="object",
                properties={
                    "directory": {
                        "type": "string",
                        "description": "Directory to list, default '.'",
                    },
                    "recursive": {"type": "boolean", "description": "Whether to list recursively"},
                },
                required=["directory"],
            ),
        ),
        Tool(
            function="read_file",
            description="Read a file with optional line range",
            parameters=ToolParameter(
                type="object",
                properties={
                    "file_path": {"type": "string", "description": "Path to the file to read"},
                    "start_line": {
                        "type": "integer",
                        "description": "Starting line number (1-based)",
                    },
                    "num_lines": {"type": "integer", "description": "Number of lines to read"},
                },
                required=["file_path"],
            ),
        ),
        Tool(
            function="run_command",
            description="Execute a shell command and return its output (include the full command string).",
            parameters=ToolParameter(
                type="object",
                properties={
                    "command": {
                        "type": "string",
                        "description": "Command to run (do not leave blank).",
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Working directory (optional; defaults to session cwd)",
                    },
                    "timeout": {"type": "number", "description": "Timeout in seconds"},
                },
                required=["command"],
            ),
        ),
        Tool(
            function="edit_file",
            description="Replace the contents of a file (requires a file_path and new_content; file_path must not be a directory).",
            parameters=ToolParameter(
                type="object",
                properties={
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to edit (relative paths are allowed).",
                    },
                    "new_content": {
                        "type": "string",
                        "description": "New content to write into the file.",
                    },
                    "create": {
                        "type": "boolean",
                        "description": "Create the file if it does not exist (default: true).",
                    },
                },
                required=["file_path", "new_content"],
            ),
        ),
        Tool(
            function="apply_patch",
            description="Apply a unified diff patch to a file",
            parameters=ToolParameter(
                type="object",
                properties={
                    "file_path": {"type": "string", "description": "Path to the file to patch"},
                    "patch": {"type": "string", "description": "Unified diff patch text"},
                    "strip": {
                        "type": "integer",
                        "description": "Strip leading path components (patch -p)",
                    },
                },
                required=["file_path", "patch"],
            ),
        ),
        Tool(
            function="file_summary",
            description="Summarize a file (head/tail and line count)",
            parameters=ToolParameter(
                type="object",
                properties={
                    "file_path": {"type": "string", "description": "Path to summarize"},
                    "head_lines": {
                        "type": "integer",
                        "description": "Number of head lines to include",
                    },
                    "tail_lines": {
                        "type": "integer",
                        "description": "Number of tail lines to include",
                    },
                },
                required=["file_path"],
            ),
        ),
        Tool(
            function="code_search",
            description="Search for a pattern in files using ripgrep",
            parameters=ToolParameter(
                type="object",
                properties={
                    "pattern": {"type": "string", "description": "Pattern to search for"},
                    "directory": {
                        "type": "string",
                        "description": "Root directory to search",
                        "default": ".",
                    },
                    "glob": {"type": "string", "description": "Glob pattern filter"},
                    "case_sensitive": {"type": "boolean", "description": "Case sensitive search"},
                    "timeout": {"type": "number", "description": "Timeout in seconds"},
                },
                required=["pattern"],
            ),
        ),
    ]
    return base_tools


async def run_tool(function_name: str, ctx: Any | None = None, **kwargs: Any) -> dict:
    handler = TOOL_HANDLERS.get(function_name)
    if not handler:
        return {"content": None, "error": f"Unknown tool function: {function_name}"}
    provided_keys = set(kwargs.keys())
    call_kwargs = dict(kwargs)
    try:
        sig = inspect.signature(handler)
        if "ctx" in sig.parameters:
            call_kwargs.setdefault("ctx", ctx)
        for name, param in sig.parameters.items():
            if name in call_kwargs:
                continue
            if param.default is not inspect._empty:
                call_kwargs[name] = param.default
            elif name != "ctx":
                call_kwargs[name] = "" if param.annotation is str or name == "command" else None
    except Exception:  # pragma: no cover - best effort filling
        if "ctx" not in call_kwargs:
            call_kwargs["ctx"] = ctx
    required = TOOL_REQUIRED_ARGS.get(function_name, [])
    if function_name != "run_command":
        missing_required = [
            name for name in required if name not in provided_keys or kwargs.get(name) in ("", None)
        ]
        if missing_required:
            return {
                "content": None,
                "error": f"Missing required arguments: {', '.join(missing_required)}",
            }
    try:
        filtered_kwargs = {k: v for k, v in call_kwargs.items() if k in sig.parameters}
        return await handler(**filtered_kwargs)
    except Exception as exc:
        # Drop unexpected args (e.g., from LLM hallucinations) and retry once
        try:
            sig = inspect.signature(handler)
            filtered = {k: v for k, v in {**kwargs, "ctx": ctx}.items() if k in sig.parameters}
            return await handler(**filtered)
        except Exception:
            return {"content": None, "error": str(exc)}


def register_readonly_tools(agent: Any) -> None:
    """Register read-only tool wrappers on the given agent (for planning delegate)."""

    for name in READ_ONLY_TOOLS:

        def _make_tool(fn_name: str):
            @agent.tool_plain(name=fn_name)  # type: ignore[misc]
            async def _wrapper(ctx: Any | None = None, **kwargs: Any) -> Any:
                return await run_tool(fn_name, ctx=ctx, **kwargs)

            return _wrapper

        _make_tool(name)
