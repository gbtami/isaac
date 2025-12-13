"""Local tool registry and ACP tool descriptions."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Type

from pydantic import ValidationError
from pydantic_ai import RunContext

from .apply_patch import apply_patch
from .code_search import code_search
from .edit_file import edit_file
from .file_summary import file_summary
from .list_files import list_files
from .read_file import read_file
from .run_command import run_command
from .args import (
    ApplyPatchArgs,
    CodeSearchArgs,
    EditFileArgs,
    FileSummaryArgs,
    ListFilesArgs,
    ReadFileArgs,
    RunCommandArgs,
)


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

# Pydantic argument models for each tool, used for schema generation and validation.
TOOL_ARG_MODELS: Dict[str, Type[Any]] = {
    "list_files": ListFilesArgs,
    "read_file": ReadFileArgs,
    "run_command": RunCommandArgs,
    "edit_file": EditFileArgs,
    "apply_patch": ApplyPatchArgs,
    "file_summary": FileSummaryArgs,
    "code_search": CodeSearchArgs,
}

TOOL_DESCRIPTIONS: Dict[str, str] = {
    "list_files": "List files and directories (defaults to current directory; set directory if you need a different root).",
    "read_file": "Read a file with optional line range",
    "run_command": "Execute a shell command and return its output (include the full command string).",
    "edit_file": "Replace the contents of a file (requires a path and content; path must not be a directory).",
    "apply_patch": "Apply a unified diff patch to a file",
    "file_summary": "Summarize a file (head/tail and line count)",
    "code_search": "Search for a pattern in files using ripgrep",
}

# Tools permitted for planning delegate (read-only, non-destructive)
READ_ONLY_TOOLS = {
    "list_files",
    "read_file",
    "file_summary",
    "code_search",
}

TOOL_REQUIRED_ARGS: Dict[str, list[str]] = {
    name: list(model.model_json_schema().get("required", []))  # type: ignore[attr-defined]
    for name, model in TOOL_ARG_MODELS.items()
}


def get_tools() -> List[Any]:
    """Return ACP tool descriptions from pydantic models."""
    base_tools: list[Any] = []
    for name in TOOL_HANDLERS.keys():
        model = TOOL_ARG_MODELS.get(name)
        if model is None:
            continue
        schema = model.model_json_schema()  # type: ignore[attr-defined]
        base_tools.append(
            Tool(
                function=name,
                description=TOOL_DESCRIPTIONS.get(name, ""),
                parameters=ToolParameter(
                    type="object",
                    properties=schema.get("properties", {}),
                    required=schema.get("required", []),
                ),
            )
        )
    return base_tools


async def run_tool(function_name: str, ctx: Any | None = None, **kwargs: Any) -> dict:
    """Run a tool by name with pydantic validation and pydantic-ai retries."""
    handler = TOOL_HANDLERS.get(function_name)
    if not handler:
        return {"content": None, "error": f"Unknown tool function: {function_name}"}

    args_model = TOOL_ARG_MODELS.get(function_name)
    required = TOOL_REQUIRED_ARGS.get(function_name, [])
    missing_required = [name for name in required if name not in kwargs or kwargs.get(name) in ("", None)]
    if missing_required:
        msg = f"Missing required arguments: {', '.join(missing_required)}"
        if ctx is not None:
            from pydantic_ai.exceptions import ToolRetryError  # type: ignore
            from pydantic_ai.messages import RetryPromptPart  # type: ignore

            raise ToolRetryError(RetryPromptPart(content=msg, tool_name=function_name))
        return {"content": None, "error": msg}

    if args_model is not None:
        try:
            parsed = args_model.model_validate(kwargs)  # type: ignore[attr-defined]
            call_kwargs: dict[str, Any] = parsed.model_dump()  # type: ignore[attr-defined]
        except ValidationError as exc:
            msg = f"Invalid arguments: {exc}"
            if ctx is not None:
                from pydantic_ai.exceptions import ToolRetryError  # type: ignore
                from pydantic_ai.messages import RetryPromptPart  # type: ignore

                raise ToolRetryError(RetryPromptPart(content=msg, tool_name=function_name))
            return {"content": None, "error": msg}
    else:
        call_kwargs = dict(kwargs)

    try:
        sig = inspect.signature(handler)
    except Exception:  # pragma: no cover - defensive
        sig = inspect.signature(lambda: None)

    if "ctx" in sig.parameters:
        call_kwargs["ctx"] = ctx
    filtered_kwargs = {k: v for k, v in call_kwargs.items() if k in sig.parameters}

    try:
        result = await handler(**filtered_kwargs)
    except Exception as exc:
        # Let pydantic-ai retry tooling errors when requested.
        try:
            from pydantic_ai.exceptions import ToolRetryError  # type: ignore

            if isinstance(exc, ToolRetryError):
                raise
        except Exception:
            pass
        # Drop unexpected args (e.g., from LLM hallucinations) and retry once.
        try:
            filtered = {k: v for k, v in {**kwargs, "ctx": ctx}.items() if k in sig.parameters}
            result = await handler(**filtered)
        except Exception:
            return {"content": None, "error": str(exc)}

    if isinstance(result, dict) and result.get("content") is None:
        result["content"] = ""
    return result


def register_readonly_tools(agent: Any) -> None:
    """Register read-only tool wrappers on the given agent (for planning delegate)."""

    @agent.tool(name="list_files")  # type: ignore[misc]
    async def list_files_tool(
        ctx: RunContext[Any],
        directory: str = ".",
        recursive: bool = True,
    ) -> Any:
        return await run_tool("list_files", ctx=ctx, directory=directory, recursive=recursive)

    @agent.tool(name="read_file")  # type: ignore[misc]
    async def read_file_tool(
        ctx: RunContext[Any],
        path: str,
        start: int | None = None,
        lines: int | None = None,
    ) -> Any:
        return await run_tool("read_file", ctx=ctx, path=path, start=start, lines=lines)

    @agent.tool(name="file_summary")  # type: ignore[misc]
    async def file_summary_tool(
        ctx: RunContext[Any],
        path: str,
        head_lines: int | None = 20,
        tail_lines: int | None = 20,
    ) -> Any:
        return await run_tool(
            "file_summary",
            ctx=ctx,
            path=path,
            head_lines=head_lines,
            tail_lines=tail_lines,
        )

    @agent.tool(name="code_search")  # type: ignore[misc]
    async def code_search_tool(
        ctx: RunContext[Any],
        pattern: str,
        directory: str = ".",
        glob: str | None = None,
        case_sensitive: bool = True,
        timeout: float | None = None,
    ) -> Any:
        return await run_tool(
            "code_search",
            ctx=ctx,
            pattern=pattern,
            directory=directory,
            glob=glob,
            case_sensitive=case_sensitive,
            timeout=timeout,
        )
