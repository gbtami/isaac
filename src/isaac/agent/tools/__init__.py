"""Local tool registry and ACP tool descriptions."""

from __future__ import annotations

import inspect
import logging
from typing import Any, Awaitable, Callable, Dict, Type

from pydantic import ValidationError
from pydantic_ai import RunContext

from .apply_patch import apply_patch
from .code_search import code_search
from .edit_file import edit_file
from .file_summary import file_summary
from .list_files import list_files
from .read_file import read_file
from .run_command import run_command
from .fetch_url import DEFAULT_FETCH_MAX_BYTES, DEFAULT_FETCH_TIMEOUT, fetch_url
from .args import (
    ApplyPatchArgs,
    CodeSearchArgs,
    EditFileArgs,
    FileSummaryArgs,
    ListFilesArgs,
    ReadFileArgs,
    RunCommandArgs,
    FetchUrlArgs,
)


ToolHandler = Callable[..., Awaitable[dict]]

TOOL_HANDLERS: Dict[str, ToolHandler] = {
    "list_files": list_files,
    "read_file": read_file,
    "run_command": run_command,
    "edit_file": edit_file,
    "code_search": code_search,
    "apply_patch": apply_patch,
    "file_summary": file_summary,
    "fetch_url": fetch_url,
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
}

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

TOOL_REQUIRED_ARGS: Dict[str, list[str]] = {
    name: list(model.model_json_schema().get("required", []))  # type: ignore[attr-defined]
    for name, model in TOOL_ARG_MODELS.items()
}


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
    _register_toolset(agent, tools=_READ_ONLY_TOOL_ORDER, logger=None)


def register_tools(agent: Any) -> None:
    """Register the full toolset on the given agent."""
    _register_toolset(agent, tools=_FULL_TOOL_ORDER, logger=logging.getLogger("acp_server"))


_FULL_TOOL_ORDER = tuple(TOOL_HANDLERS.keys())
_READ_ONLY_TOOL_ORDER = tuple(name for name in _FULL_TOOL_ORDER if name in READ_ONLY_TOOLS)


class _ToolRegistrar:
    def __init__(self, logger: logging.Logger | None) -> None:
        self._logger = logger

    def _log(self, tool_name: str, args: dict[str, Any]) -> None:
        if self._logger is None:
            return
        self._logger.info("Pydantic tool invoked: %s args=%s", tool_name, args)

    async def list_files_tool(
        self,
        ctx: RunContext[Any],
        directory: str = ".",
        recursive: bool = True,
    ) -> Any:
        self._log("list_files", {"directory": directory, "recursive": recursive})
        return await run_tool("list_files", ctx=ctx, directory=directory, recursive=recursive)

    async def read_file_tool(
        self,
        ctx: RunContext[Any],
        path: str,
        start: int | None = None,
        lines: int | None = None,
    ) -> Any:
        self._log("read_file", {"path": path, "start": start, "lines": lines})
        return await run_tool("read_file", ctx=ctx, path=path, start=start, lines=lines)

    async def run_command_tool(
        self,
        ctx: RunContext[Any],
        command: str,
        cwd: str | None = None,
        timeout: float | None = None,
    ) -> Any:
        self._log("run_command", {"command": command, "cwd": cwd, "timeout": timeout})
        return await run_tool("run_command", ctx=ctx, command=command, cwd=cwd, timeout=timeout)

    async def edit_file_tool(
        self,
        ctx: RunContext[Any],
        path: str,
        content: str,
        create: bool = True,
    ) -> Any:
        self._log("edit_file", {"path": path, "create": create})
        return await run_tool("edit_file", ctx=ctx, path=path, content=content, create=create)

    async def apply_patch_tool(
        self,
        ctx: RunContext[Any],
        path: str,
        patch: str,
        strip: int | None = None,
    ) -> Any:
        self._log("apply_patch", {"path": path, "strip": strip})
        return await run_tool("apply_patch", ctx=ctx, path=path, patch=patch, strip=strip)

    async def file_summary_tool(
        self,
        ctx: RunContext[Any],
        path: str,
        head_lines: int | None = 20,
        tail_lines: int | None = 20,
    ) -> Any:
        self._log("file_summary", {"path": path, "head_lines": head_lines, "tail_lines": tail_lines})
        return await run_tool(
            "file_summary",
            ctx=ctx,
            path=path,
            head_lines=head_lines,
            tail_lines=tail_lines,
        )

    async def code_search_tool(
        self,
        ctx: RunContext[Any],
        pattern: str,
        directory: str = ".",
        glob: str | None = None,
        case_sensitive: bool = True,
        timeout: float | None = None,
    ) -> Any:
        self._log(
            "code_search",
            {
                "pattern": pattern,
                "directory": directory,
                "glob": glob,
                "case_sensitive": case_sensitive,
                "timeout": timeout,
            },
        )
        return await run_tool(
            "code_search",
            ctx=ctx,
            pattern=pattern,
            directory=directory,
            glob=glob,
            case_sensitive=case_sensitive,
            timeout=timeout,
        )

    async def fetch_url_tool(
        self,
        ctx: RunContext[Any],
        url: str,
        max_bytes: int = DEFAULT_FETCH_MAX_BYTES,
        timeout: float | None = DEFAULT_FETCH_TIMEOUT,
    ) -> Any:
        self._log("fetch_url", {"url": url, "max_bytes": max_bytes, "timeout": timeout})
        return await run_tool("fetch_url", ctx=ctx, url=url, max_bytes=max_bytes, timeout=timeout)


def _register_toolset(
    agent: Any,
    *,
    tools: tuple[str, ...],
    logger: logging.Logger | None,
) -> None:
    registrar = _ToolRegistrar(logger)
    tool_map = {
        "list_files": (registrar.list_files_tool, DEFAULT_TOOL_TIMEOUT_S),
        "read_file": (registrar.read_file_tool, DEFAULT_TOOL_TIMEOUT_S),
        "run_command": (registrar.run_command_tool, RUN_COMMAND_TIMEOUT_S),
        "edit_file": (registrar.edit_file_tool, DEFAULT_TOOL_TIMEOUT_S),
        "apply_patch": (registrar.apply_patch_tool, DEFAULT_TOOL_TIMEOUT_S),
        "file_summary": (registrar.file_summary_tool, DEFAULT_TOOL_TIMEOUT_S),
        "code_search": (registrar.code_search_tool, DEFAULT_TOOL_TIMEOUT_S),
        "fetch_url": (registrar.fetch_url_tool, DEFAULT_FETCH_TIMEOUT),
    }
    for name in tools:
        func, timeout = tool_map[name]
        agent.tool(name=name, timeout=timeout)(func)  # type: ignore[misc]
