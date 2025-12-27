"""Tool execution helpers for ACP agent prompt/tool calls."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict

from acp import (
    CreateTerminalRequest,
    KillTerminalCommandRequest,
    ReleaseTerminalRequest,
    TerminalOutputRequest,
)
from acp.helpers import session_notification, text_block, tool_content, tool_diff_content
from acp.schema import SessionNotification
from acp.contrib.tool_calls import ToolCallTracker

from isaac.agent.agent_terminal import (
    TerminalState,
    create_terminal,
    kill_terminal,
    release_terminal,
    terminal_output,
)
from isaac.agent.tool_io import await_with_cancel, truncate_text, truncate_tool_output
from isaac.agent.tools import run_tool
from isaac.log_utils import log_context, log_event

logger = logging.getLogger(__name__)


@dataclass
class ToolExecutionContext:
    send_update: Callable[[SessionNotification], Awaitable[None]]
    request_run_permission: Callable[[str, str, str, str | None], Awaitable[bool]]
    session_cwds: Dict[str, Path]
    session_modes: Dict[str, str]
    terminals: Dict[str, TerminalState]
    cancel_events: Dict[str, asyncio.Event]
    tool_output_limit: int
    terminal_output_limit: int
    command_timeout_s: float


async def execute_tool(
    ctx: ToolExecutionContext,
    session_id: str,
    *,
    tool_name: str,
    tool_call_id: str | None = None,
    arguments: dict[str, Any] | None = None,
) -> None:
    """Execute a regular tool and stream ACP tool_call_update notifications."""
    tool_call_id = tool_call_id or str(uuid.uuid4())
    tracker = ToolCallTracker(id_factory=lambda: tool_call_id)
    with log_context(session_id=session_id, tool_call_id=tool_call_id, tool_name=tool_name):
        log_event(logger, "tool.call.start", args_keys=sorted(arguments or {}))
    start = tracker.start(
        external_id=tool_call_id,
        title=tool_name,
        status="in_progress",
        raw_input={"tool": tool_name, **(arguments or {})},
    )
    await ctx.send_update(session_notification(session_id, start))

    cancel_event = ctx.cancel_events.setdefault(session_id, asyncio.Event())
    result: dict[str, Any] | None = await await_with_cancel(
        run_tool(
            tool_name,
            cwd=str(ctx.session_cwds.get(session_id, Path.cwd())),
            **(arguments or {}),
        ),
        cancel_event,
    )
    if result is None:
        with log_context(session_id=session_id, tool_call_id=tool_call_id, tool_name=tool_name):
            log_event(logger, "tool.call.cancelled")
        progress = tracker.progress(
            external_id=tool_call_id,
            status="failed",
            raw_output={"content": None, "error": "cancelled"},
            content=[tool_content(text_block("Cancelled"))],
        )
        await ctx.send_update(session_notification(session_id, progress))
        return

    # Include tool name for downstream clients to interpret plan tool output.
    result_with_tool = dict(result)
    result_with_tool.setdefault("tool", tool_name)
    result_with_tool, was_truncated = truncate_tool_output(result_with_tool, ctx.tool_output_limit)
    if was_truncated:
        result_with_tool["truncated"] = True
    status = "completed" if not result_with_tool.get("error") else "failed"
    summary = result_with_tool.get("content") or result_with_tool.get("error") or ""
    content_blocks: list[Any] = []
    if tool_name == "edit_file":
        new_text = result_with_tool.get("new_text")
        old_text = result_with_tool.get("old_text")
        path = (arguments or {}).get("path", "")
        if isinstance(new_text, str):
            with contextlib.suppress(Exception):
                content_blocks.append(tool_diff_content(path, new_text, old_text))
    if not content_blocks:
        content_blocks = [tool_content(text_block(summary))]
    with log_context(session_id=session_id, tool_call_id=tool_call_id, tool_name=tool_name):
        log_event(
            logger,
            "tool.call.complete",
            status=status,
            summary_preview=str(summary)[:160].replace("\n", "\\n"),
        )
    progress = tracker.progress(
        external_id=tool_call_id,
        status=status,
        raw_output=result_with_tool,
        content=content_blocks,
    )
    await ctx.send_update(session_notification(session_id, progress))


async def execute_run_command_with_terminal(
    ctx: ToolExecutionContext,
    session_id: str,
    *,
    tool_call_id: str,
    arguments: dict[str, Any],
) -> None:
    """Run the run_command tool using an ACP terminal for streamed output."""
    tracker = ToolCallTracker(id_factory=lambda: tool_call_id)
    command = arguments.get("command") or ""
    cwd_arg = arguments.get("cwd")
    with log_context(session_id=session_id, tool_call_id=tool_call_id, tool_name="run_command"):
        log_event(logger, "tool.run_command.start", command=command.strip(), cwd=cwd_arg)

    start = tracker.start(
        external_id=tool_call_id,
        title="run_command",
        status="in_progress",
        raw_input={"tool": "run_command", **arguments},
    )
    await ctx.send_update(session_notification(session_id, start))

    # Request permission before executing shell commands (ask mode only).
    mode = ctx.session_modes.get(session_id, "ask")
    if mode == "ask":
        allowed = await ctx.request_run_permission(
            session_id=session_id,
            tool_call_id=tool_call_id,
            command=command,
            cwd=cwd_arg,
        )
        if not allowed:
            with log_context(session_id=session_id, tool_call_id=tool_call_id, tool_name="run_command"):
                log_event(logger, "tool.run_command.denied")
            progress = tracker.progress(
                external_id=tool_call_id,
                status="failed",
                raw_output={"content": None, "error": "permission denied"},
                content=[tool_content(text_block("Command blocked: permission denied"))],
            )
            await ctx.send_update(session_notification(session_id, progress))
            return

    cancel_event = ctx.cancel_events.setdefault(session_id, asyncio.Event())
    cancel_event.clear()
    timeout_s = float(arguments.get("timeout") or ctx.command_timeout_s)
    with log_context(session_id=session_id, tool_call_id=tool_call_id, tool_name="run_command"):
        log_event(logger, "tool.run_command.config", timeout_s=timeout_s)

    try:
        create_resp = await create_terminal(
            ctx.session_cwds,
            ctx.terminals,
            CreateTerminalRequest(
                session_id=session_id,
                command="bash",
                args=["-lc", command],
                cwd=cwd_arg,
                output_byte_limit=ctx.terminal_output_limit,
            ),
        )
    except Exception as exc:  # pragma: no cover - defensive
        with log_context(session_id=session_id, tool_call_id=tool_call_id, tool_name="run_command"):
            log_event(logger, "tool.run_command.error", level=logging.WARNING, error=str(exc))
        progress = tracker.progress(
            external_id=tool_call_id,
            status="failed",
            raw_output={"content": None, "error": f"Failed to start command: {exc}"},
            content=[tool_content(text_block(f"Failed to start command: {exc}"))],
        )
        await ctx.send_update(session_notification(session_id, progress))
        return

    term_id = create_resp.terminal_id
    collected: list[str] = []
    truncated = False
    exit_code: int | None = None
    error_msg: str | None = None

    try:
        start_time = asyncio.get_event_loop().time()
        while True:
            if cancel_event.is_set():
                error_msg = "cancelled"
                await kill_terminal(
                    ctx.terminals,
                    KillTerminalCommandRequest(session_id=session_id, terminal_id=term_id),
                )
                break

            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout_s:
                error_msg = f"Command timed out after {timeout_s}s"
                await kill_terminal(
                    ctx.terminals,
                    KillTerminalCommandRequest(session_id=session_id, terminal_id=term_id),
                )
                break

            out_resp = await terminal_output(
                ctx.terminals,
                TerminalOutputRequest(session_id=session_id, terminal_id=term_id),
            )
            chunk = out_resp.output or ""
            if chunk:
                chunk, chunk_truncated = truncate_text(chunk, ctx.tool_output_limit)
                collected.append(chunk)
                truncated = truncated or out_resp.truncated or chunk_truncated
                progress = tracker.progress(
                    external_id=tool_call_id,
                    status="in_progress",
                    raw_output={
                        "content": chunk,
                        "error": None,
                        "returncode": exit_code,
                        "truncated": out_resp.truncated or chunk_truncated,
                    },
                    content=[tool_content(text_block(chunk))],
                )
                await ctx.send_update(session_notification(session_id, progress))

            if out_resp.exit_status:
                exit_code = out_resp.exit_status.exit_code
                break

            await asyncio.sleep(0.2)
    finally:
        with contextlib.suppress(Exception):
            await release_terminal(
                ctx.terminals,
                ReleaseTerminalRequest(session_id=session_id, terminal_id=term_id),
            )

    if not collected:
        with contextlib.suppress(Exception):
            final_out = await terminal_output(
                ctx.terminals,
                TerminalOutputRequest(session_id=session_id, terminal_id=term_id),
            )
            if final_out.output:
                capped_output, chunk_truncated = truncate_text(final_out.output, ctx.tool_output_limit)
                collected.append(capped_output)
                truncated = truncated or final_out.truncated or chunk_truncated
                exit_code = exit_code or (final_out.exit_status.exit_code if final_out.exit_status else None)

    full_output, capped = truncate_text("".join(collected).rstrip("\n"), ctx.tool_output_limit)
    truncated = truncated or capped
    status = "failed" if error_msg else "completed"
    if exit_code not in (0, None) and not error_msg:
        status = "failed"

    summary = error_msg or full_output or ""
    progress = tracker.progress(
        external_id=tool_call_id,
        status=status,
        raw_output={
            "content": full_output,
            "error": error_msg,
            "returncode": exit_code,
            "truncated": truncated,
        },
        content=[tool_content(text_block(summary))] if summary else None,
    )
    await ctx.send_update(session_notification(session_id, progress))
    with log_context(session_id=session_id, tool_call_id=tool_call_id, tool_name="run_command"):
        log_event(
            logger,
            "tool.run_command.complete",
            status=status,
            returncode=exit_code,
            truncated=truncated,
            error=error_msg,
        )
