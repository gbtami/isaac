"""Register tool handlers with pydantic-ai agents."""

from __future__ import annotations

import logging
from typing import Any, Iterable

from pydantic_ai import RunContext

from isaac.agent.subagents import DELEGATE_TOOL_TIMEOUTS
from isaac.agent.tools.executor import run_tool
from isaac.agent.tools.registry import (
    DEFAULT_FETCH_MAX_BYTES,
    DEFAULT_FETCH_TIMEOUT,
    DEFAULT_TOOL_TIMEOUT_S,
    RUN_COMMAND_TIMEOUT_S,
    _FULL_TOOL_ORDER,
    _READ_ONLY_TOOL_ORDER,
)


def register_readonly_tools(agent: Any, *, tool_names: Iterable[str] | None = None) -> None:
    """Register read-only tool wrappers on the given agent (for planning delegate)."""
    tools = _READ_ONLY_TOOL_ORDER if tool_names is None else tuple(tool_names)
    _register_toolset(agent, tools=tools, logger=None)


def register_tools(agent: Any, *, tool_names: Iterable[str] | None = None) -> None:
    """Register the toolset on the given agent."""
    tools = _FULL_TOOL_ORDER if tool_names is None else tuple(tool_names)
    _register_toolset(agent, tools=tools, logger=logging.getLogger("acp_server"))


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
        "planner": (registrar.planner_tool, DELEGATE_TOOL_TIMEOUTS.get("planner", DEFAULT_TOOL_TIMEOUT_S)),
        "review": (registrar.review_tool, DELEGATE_TOOL_TIMEOUTS.get("review", DEFAULT_TOOL_TIMEOUT_S)),
        "coding": (registrar.coding_tool, DELEGATE_TOOL_TIMEOUTS.get("coding", DEFAULT_TOOL_TIMEOUT_S)),
    }
    for name in tools:
        func, timeout = tool_map[name]
        agent.tool(name=name, timeout=timeout)(func)  # type: ignore[misc]


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

    async def planner_tool(
        self,
        ctx: RunContext[Any],
        task: str,
        context: str | None = None,
    ) -> Any:
        self._log("planner", {"task": task, "context": context})
        return await run_tool("planner", ctx=ctx, task=task, context=context)

    async def review_tool(
        self,
        ctx: RunContext[Any],
        task: str,
        context: str | None = None,
    ) -> Any:
        self._log("review", {"task": task, "context": context})
        return await run_tool("review", ctx=ctx, task=task, context=context)

    async def coding_tool(
        self,
        ctx: RunContext[Any],
        task: str,
        context: str | None = None,
    ) -> Any:
        self._log("coding", {"task": task, "context": context})
        return await run_tool("coding", ctx=ctx, task=task, context=context)
