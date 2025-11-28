# This file makes 'tools' a Python package
# It also exports the available ACP tools

from typing import List
from acp.schema import Tool, ToolParameter

def get_tools() -> List[Tool]:
    """Return the list of available ACP tools."""
    return [
        Tool(
            function="tool_list_files",
            description="List files and directories recursively",
            parameters=ToolParameter(
                type="object",
                properties={
                    "directory": {"type": "string", "description": "Directory to list, default '.'"},
                    "recursive": {"type": "boolean", "description": "Whether to list recursively", "default": True}
                },
                required=["directory"]
            )
        ),
        Tool(
            function="tool_read_file",
            description="Read a file with optional line range",
            parameters=ToolParameter(
                type="object",
                properties={
                    "file_path": {"type": "string", "description": "Path to the file to read"},
                    "start_line": {"type": "integer", "description": "Starting line number (1-based)"},
                    "num_lines": {"type": "integer", "description": "Number of lines to read"}
                },
                required=["file_path"]
            )
        )
    ]
