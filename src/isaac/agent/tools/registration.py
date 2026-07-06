"""Build Isaac tool handlers as Pydantic AI toolsets and capabilities."""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Iterable

from pydantic_ai import FunctionToolset, Tool  # type: ignore
from pydantic_ai.capabilities import Toolset as ToolsetCapability  # type: ignore

from isaac.agent.ai_types import SessionToolDeps, ToolContext

from isaac.agent.subagents import DELEGATE_TOOL_TIMEOUTS
from isaac.agent.tools.executor import run_tool
from isaac.agent.tools.registry import (
    DEFAULT_FETCH_MAX_BYTES,
    DEFAULT_FETCH_TIMEOUT,
    DEFAULT_TOOL_TIMEOUT_S,
    RUN_COMMAND_TIMEOUT_S,
    TOOL_DESCRIPTIONS,
    _FULL_TOOL_ORDER,
)
from isaac.log_utils import log_event

_MUTATING_TOOLS = {"run_command", "edit_file", "apply_patch"}


def build_isaac_toolset(
    *,
    tool_names: Iterable[str] | None = None,
    logger: logging.Logger | None = None,
    toolset_id: str = "isaac-core-tools",
) -> FunctionToolset[Any]:
    """Build Isaac's ACP-compatible Pydantic AI function toolset.

    Tool names, argument schemas, timeouts, approval requirements, and metadata
    define the ACP-visible Isaac coding tool contract.
    """

    tools = _FULL_TOOL_ORDER if tool_names is None else tuple(tool_names)
    registrar = _ToolRegistrar(logger)
    toolset: FunctionToolset[Any] = FunctionToolset(id=toolset_id)
    for name in tools:
        func, timeout = _tool_function(registrar, name)
        toolset.add_tool(_build_tool(name=name, func=func, timeout=timeout))
    return toolset


def build_isaac_tools_capability(
    *,
    tool_names: Iterable[str] | None = None,
    logger: logging.Logger | None = None,
    capability_id: str = "isaac-core-tools",
) -> Any:
    """Build the capability that contributes Isaac's core coding tools."""

    return ToolsetCapability(
        build_isaac_toolset(
            tool_names=tool_names,
            logger=logger,
            toolset_id=capability_id,
        )
    )


def _build_tool(name: str, func: Callable[..., Awaitable[Any]], timeout: float | None) -> Tool[Any]:
    """Create a Pydantic AI Tool preserving Isaac's ACP-visible metadata."""

    is_delegate = name in DELEGATE_TOOL_TIMEOUTS
    metadata = {
        "tool_group": "delegate" if is_delegate else "core",
        "mutates_state": name in _MUTATING_TOOLS,
    }
    return Tool(
        func,
        takes_ctx=True,
        name=name,
        description=TOOL_DESCRIPTIONS.get(name),
        timeout=timeout,
        strict=True,
        sequential=(name in _MUTATING_TOOLS) or is_delegate,
        requires_approval=name == "run_command",
        metadata=metadata,
    )


def _tool_function(registrar: "_ToolRegistrar", name: str) -> tuple[Callable[..., Awaitable[Any]], float | None]:
    """Resolve a tool wrapper and timeout by public Isaac tool name."""

    tool_map: dict[str, tuple[Callable[..., Awaitable[Any]], float | None]] = {
        "list_files": (registrar.list_files_tool, DEFAULT_TOOL_TIMEOUT_S),
        "read_file": (registrar.read_file_tool, DEFAULT_TOOL_TIMEOUT_S),
        "run_command": (registrar.run_command_tool, RUN_COMMAND_TIMEOUT_S),
        "edit_file": (registrar.edit_file_tool, DEFAULT_TOOL_TIMEOUT_S),
        "apply_patch": (registrar.apply_patch_tool, DEFAULT_TOOL_TIMEOUT_S),
        "file_summary": (registrar.file_summary_tool, DEFAULT_TOOL_TIMEOUT_S),
        "code_search": (registrar.code_search_tool, DEFAULT_TOOL_TIMEOUT_S),
        "fetch_url": (registrar.fetch_url_tool, DEFAULT_FETCH_TIMEOUT),
    }
    for delegate_name, timeout in DELEGATE_TOOL_TIMEOUTS.items():
        tool_map[delegate_name] = (registrar.delegate_tool(delegate_name), timeout)
    return tool_map[name]


class _ToolRegistrar:
    """Builds tool wrappers that route calls through run_tool.

    Wrappers handle logging and keep the pydantic-ai interface consistent.
    """

    def __init__(self, logger: logging.Logger | None) -> None:
        self._logger = logger

    def _log(self, tool_name: str, args: dict[str, Any]) -> None:
        """Emit a debug log for tool invocation arguments."""
        if self._logger is None:
            return
        log_event(self._logger, "tool.wrapper.invoke", level=logging.DEBUG, tool=tool_name, args=args)

    @staticmethod
    def _runtime_kwargs(ctx: ToolContext) -> dict[str, Any]:
        """Return session-bound runtime kwargs hidden from model schemas."""

        deps = getattr(ctx, "deps", None)
        if not isinstance(deps, SessionToolDeps):
            return {}
        return {
            "session_cwd": str(deps.cwd),
            "additional_directories": tuple(str(path) for path in deps.additional_directories),
        }

    def delegate_tool(
        self, tool_name: str
    ) -> Callable[[ToolContext, str, str | None, str | None, bool], Awaitable[Any]]:
        """Build a wrapper for delegate tools that share the base delegate args.

        Delegate tools always accept task/context/session/carryover to keep the
        tool call contract uniform across sub-agents.
        """

        async def _tool(
            ctx: ToolContext,
            task: str,
            context: str | None = None,
            session_id: str | None = None,
            carryover: bool = False,
        ) -> Any:
            self._log(
                tool_name,
                {
                    "task": task,
                    "context": context,
                    "session_id": session_id,
                    "carryover": carryover,
                },
            )
            return await run_tool(
                tool_name,
                ctx=ctx,
                **self._runtime_kwargs(ctx),
                task=task,
                context=context,
                session_id=session_id,
                carryover=carryover,
            )

        return _tool

    async def list_files_tool(
        self,
        ctx: ToolContext,
        directory: str = ".",
        recursive: bool = True,
    ) -> Any:
        self._log("list_files", {"directory": directory, "recursive": recursive})
        return await run_tool(
            "list_files",
            ctx=ctx,
            **self._runtime_kwargs(ctx),
            directory=directory,
            recursive=recursive,
        )

    async def read_file_tool(
        self,
        ctx: ToolContext,
        path: str,
        start: int | None = None,
        lines: int | None = None,
    ) -> Any:
        self._log("read_file", {"path": path, "start": start, "lines": lines})
        return await run_tool("read_file", ctx=ctx, **self._runtime_kwargs(ctx), path=path, start=start, lines=lines)

    async def run_command_tool(
        self,
        ctx: ToolContext,
        command: str,
        cwd: str | None = None,
        timeout: float | None = None,
    ) -> Any:
        self._log("run_command", {"command": command, "cwd": cwd, "timeout": timeout})
        return await run_tool(
            "run_command",
            ctx=ctx,
            **self._runtime_kwargs(ctx),
            command=command,
            cwd=cwd,
            timeout=timeout,
        )

    async def edit_file_tool(
        self,
        ctx: ToolContext,
        path: str,
        content: str,
        create: bool = True,
        expected_sha256: str | None = None,
    ) -> Any:
        self._log("edit_file", {"path": path, "create": create, "expected_sha256": expected_sha256})
        return await run_tool(
            "edit_file",
            ctx=ctx,
            **self._runtime_kwargs(ctx),
            path=path,
            content=content,
            create=create,
            expected_sha256=expected_sha256,
        )

    async def apply_patch_tool(
        self,
        ctx: ToolContext,
        path: str,
        patch: str,
        strip: int | None = None,
        expected_sha256: str | None = None,
    ) -> Any:
        self._log("apply_patch", {"path": path, "strip": strip, "expected_sha256": expected_sha256})
        return await run_tool(
            "apply_patch",
            ctx=ctx,
            **self._runtime_kwargs(ctx),
            path=path,
            patch=patch,
            strip=strip,
            expected_sha256=expected_sha256,
        )

    async def file_summary_tool(
        self,
        ctx: ToolContext,
        path: str,
        head_lines: int | None = 20,
        tail_lines: int | None = 20,
    ) -> Any:
        self._log("file_summary", {"path": path, "head_lines": head_lines, "tail_lines": tail_lines})
        return await run_tool(
            "file_summary",
            ctx=ctx,
            **self._runtime_kwargs(ctx),
            path=path,
            head_lines=head_lines,
            tail_lines=tail_lines,
        )

    async def code_search_tool(
        self,
        ctx: ToolContext,
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
            **self._runtime_kwargs(ctx),
            pattern=pattern,
            directory=directory,
            glob=glob,
            case_sensitive=case_sensitive,
            timeout=timeout,
        )

    async def fetch_url_tool(
        self,
        ctx: ToolContext,
        url: str,
        max_bytes: int = DEFAULT_FETCH_MAX_BYTES,
        timeout: float | None = DEFAULT_FETCH_TIMEOUT,
    ) -> Any:
        self._log("fetch_url", {"url": url, "max_bytes": max_bytes, "timeout": timeout})
        return await run_tool("fetch_url", ctx=ctx, url=url, max_bytes=max_bytes, timeout=timeout)
