"""Local tool registry and ACP tool descriptions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List

from .list_directory import list_files
from .read_file import read_file
from .run_command import run_command
from .edit_file import edit_file
from .code_search import code_search

try:
    from acp.schema import Tool, ToolParameter  # type: ignore
except Exception:  # pragma: no cover - fallback for older SDKs

    @dataclass
    class ToolParameter:  # type: ignore[misc]
        type: str
        properties: dict[str, Any]
        required: list[str]

    @dataclass
    class Tool:  # type: ignore[misc]
        function: str
        description: str
        parameters: ToolParameter


ToolHandler = Callable[..., Awaitable[dict]]

TOOL_HANDLERS: Dict[str, ToolHandler] = {
    "tool_list_files": list_files,
    "tool_read_file": read_file,
    "tool_run_command": run_command,
    "tool_edit_file": edit_file,
    "tool_code_search": code_search,
}


def get_tools() -> List[Any]:
    """Return ACP tool descriptions (with graceful fallback when schema lacks Tool)."""
    return [
        Tool(
            function="tool_list_files",
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
            function="tool_read_file",
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
            function="tool_run_command",
            description="Execute a shell command and return its output",
            parameters=ToolParameter(
                type="object",
                properties={
                    "command": {"type": "string", "description": "Command to run"},
                    "cwd": {"type": "string", "description": "Working directory"},
                    "timeout": {"type": "number", "description": "Timeout in seconds"},
                },
                required=["command"],
            ),
        ),
        Tool(
            function="tool_edit_file",
            description="Replace the contents of a file",
            parameters=ToolParameter(
                type="object",
                properties={
                    "file_path": {"type": "string", "description": "Path to the file to edit"},
                    "new_content": {"type": "string", "description": "New content to write"},
                    "create": {"type": "boolean", "description": "Create file if missing"},
                },
                required=["file_path", "new_content"],
            ),
        ),
        Tool(
            function="tool_code_search",
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


async def run_tool(function_name: str, **kwargs: Any) -> dict:
    handler = TOOL_HANDLERS.get(function_name)
    if not handler:
        return {"content": None, "error": f"Unknown tool function: {function_name}"}
    try:
        return await handler(**kwargs)
    except Exception as exc:
        # Drop unexpected args (e.g., from LLM hallucinations) and retry once
        try:
            import inspect

            sig = inspect.signature(handler)
            filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
            return await handler(**filtered)
        except Exception:
            return {"content": None, "error": str(exc)}


# DSL parsing for "tool:<name> ..." prompts --------------------------


def _parse_list_files(args: list[str]) -> dict | None:
    directory = args[0] if args else "."
    return {"directory": directory, "recursive": True}


def _parse_read_file(args: list[str]) -> dict | None:
    if not args:
        return None
    file_path = args[0]
    start_line = int(args[1]) if len(args) > 1 else None
    num_lines = int(args[2]) if len(args) > 2 else None
    return {"file_path": file_path, "start_line": start_line, "num_lines": num_lines}


def _parse_run_command(raw: str) -> dict | None:
    cmd = raw.strip()
    if not cmd:
        return None
    return {"command": cmd}


def _parse_edit_file(args: list[str]) -> dict | None:
    if not args:
        return None
    file_path = args[0]
    new_content = " ".join(args[1:]) if len(args) > 1 else ""
    return {"file_path": file_path, "new_content": new_content}


def _parse_code_search(args: list[str]) -> dict | None:
    if not args:
        return None
    pattern = args[0]
    directory = args[1] if len(args) > 1 else "."
    return {"pattern": pattern, "directory": directory}


DSL_PARSERS: Dict[str, Callable[[list[str], str], dict | None]] = {
    "list_files": lambda args, raw: _parse_list_files(args),
    "read_file": lambda args, raw: _parse_read_file(args),
    "run_command": lambda args, raw: _parse_run_command(raw),
    "edit_file": lambda args, raw: _parse_edit_file(args),
    "code_search": lambda args, raw: _parse_code_search(args),
}

DSL_TO_TOOL: Dict[str, str] = {
    "list_files": "tool_list_files",
    "read_file": "tool_read_file",
    "run_command": "tool_run_command",
    "edit_file": "tool_edit_file",
    "code_search": "tool_code_search",
}


def parse_tool_request(prompt_text: str) -> dict[str, Any] | None:
    if not prompt_text.startswith("tool:"):
        return None

    raw = prompt_text[len("tool:") :].strip()
    if not raw:
        return None

    parts = raw.split()
    if not parts:
        return None

    dsl_name = parts[0]
    args = parts[1:]
    parser = DSL_PARSERS.get(dsl_name)
    tool_name = DSL_TO_TOOL.get(dsl_name)
    if not parser or not tool_name:
        return None

    parsed_args = parser(args, raw[len(dsl_name) :].strip() if raw.startswith(dsl_name) else raw)
    if parsed_args is None:
        return None
    return {"tool_name": tool_name, **parsed_args}
