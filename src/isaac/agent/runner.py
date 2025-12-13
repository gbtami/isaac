"""Shared runner utilities for pydantic-ai models and tool registration."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import json
from typing import Any, Callable

from pydantic_ai import RunContext
from pydantic_ai import exceptions as ai_exc  # type: ignore

from pydantic_ai.messages import (  # type: ignore
    PartDeltaEvent,
    PartEndEvent,
    TextPartDelta,
    ThinkingPartDelta,
)
from pydantic_ai.run import AgentRunResultEvent  # type: ignore

from isaac.agent.tools import run_tool

HISTORY_LOG_MAX = 8000


def register_tools(agent: Any) -> None:
    logger = logging.getLogger("acp_server")

    @agent.tool(name="list_files")  # type: ignore[misc]
    async def list_files_tool(
        ctx: RunContext[Any],
        directory: str = ".",
        recursive: bool = True,
    ) -> Any:
        logger.info(
            "Pydantic tool invoked: list_files args=%s",
            {"directory": directory, "recursive": recursive},
        )
        return await run_tool("list_files", ctx=ctx, directory=directory, recursive=recursive)

    @agent.tool(name="read_file")  # type: ignore[misc]
    async def read_file_tool(
        ctx: RunContext[Any],
        path: str,
        start: int | None = None,
        lines: int | None = None,
    ) -> Any:
        logger.info(
            "Pydantic tool invoked: read_file args=%s",
            {"path": path, "start": start, "lines": lines},
        )
        return await run_tool("read_file", ctx=ctx, path=path, start=start, lines=lines)

    @agent.tool(name="run_command")  # type: ignore[misc]
    async def run_command_tool(
        ctx: RunContext[Any],
        command: str,
        cwd: str | None = None,
        timeout: float | None = None,
    ) -> Any:
        logger.info(
            "Pydantic tool invoked: run_command args=%s",
            {"command": command, "cwd": cwd, "timeout": timeout},
        )
        return await run_tool("run_command", ctx=ctx, command=command, cwd=cwd, timeout=timeout)

    @agent.tool(name="edit_file")  # type: ignore[misc]
    async def edit_file_tool(
        ctx: RunContext[Any],
        path: str,
        content: str,
        create: bool = True,
    ) -> Any:
        logger.info(
            "Pydantic tool invoked: edit_file args=%s",
            {"path": path, "create": create},
        )
        return await run_tool("edit_file", ctx=ctx, path=path, content=content, create=create)

    @agent.tool(name="apply_patch")  # type: ignore[misc]
    async def apply_patch_tool(
        ctx: RunContext[Any],
        path: str,
        patch: str,
        strip: int | None = None,
    ) -> Any:
        logger.info(
            "Pydantic tool invoked: apply_patch args=%s",
            {"path": path, "strip": strip},
        )
        return await run_tool("apply_patch", ctx=ctx, path=path, patch=patch, strip=strip)

    @agent.tool(name="file_summary")  # type: ignore[misc]
    async def file_summary_tool(
        ctx: RunContext[Any],
        path: str,
        head_lines: int | None = 20,
        tail_lines: int | None = 20,
    ) -> Any:
        logger.info(
            "Pydantic tool invoked: file_summary args=%s",
            {"path": path, "head_lines": head_lines, "tail_lines": tail_lines},
        )
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
        logger.info(
            "Pydantic tool invoked: code_search args=%s",
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


async def stream_with_runner(
    runner: Any,
    prompt_text: str,
    on_text: Callable[[str], asyncio.Future | Any],
    on_thought: Callable[[str], asyncio.Future | Any] | None = None,
    cancel_event: asyncio.Event | None = None,
    *,
    history: Any | None = None,
    on_event: Callable[[Any], asyncio.Future | bool | None] | None = None,
    store_messages: Callable[[Any], None] | None = None,
) -> tuple[str | None, Any | None]:
    """Stream responses using the runner's streaming API.

    `on_event` lets callers react to tool call events (used to emit ACP tool updates).
    """
    stream_logger = logging.getLogger("acp_server")
    llm_logger = logging.getLogger("isaac.llm")
    stream_logger.info("LLM stream start prompt_preview=%s", prompt_text[:200])
    llm_logger.info("SEND prompt\n%s", prompt_text)
    cancel_event = cancel_event or asyncio.Event()

    output_parts: list[str] = []
    usage = None

    def _sanitize_history(raw_history: Any) -> Any:
        """Drop invalid/empty history messages to avoid provider 400s."""

        cleaned = []
        if isinstance(raw_history, list):
            for msg in raw_history:
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role")
                content = msg.get("content")
                if not role or content is None:
                    continue
                content_str = str(content).strip()
                if content_str == "":
                    continue
                cleaned.append({"role": role, "content": content_str})
        return cleaned if cleaned else None

    def _inject_system_prompt(raw_history: list[dict[str, str]] | None, runner_obj: Any) -> list[dict[str, str]] | None:
        """Ensure the system prompt is present when passing history to the model.

        pydantic-ai skips generating the system prompt when message_history is provided;
        we always inject the runner's system prompt (replacing any existing system
        messages) to preserve behavior across turns.
        """

        if not raw_history:
            return raw_history
        # Prefer stored system prompts on the pydantic-ai Agent.
        sys_prompt = None
        try:
            prompts = getattr(runner_obj, "_system_prompts", None)
            if prompts:
                sys_prompt = "\n".join([p for p in prompts if p])
        except Exception:
            pass
        if not sys_prompt:
            sys_prompt = getattr(runner_obj, "system_prompt", None)
        if callable(sys_prompt):
            try:
                sys_prompt = sys_prompt()
            except Exception:
                sys_prompt = None
        if not sys_prompt:
            return raw_history
        non_system = [msg for msg in raw_history if msg.get("role") != "system"]
        return [{"role": "system", "content": str(sys_prompt)}] + non_system

    def _convert_to_model_messages(raw_history: list[dict[str, str]] | None) -> list[Any] | None:
        """Pass through sanitized history; pydantic-ai accepts role/content dicts."""

        return raw_history

    try:
        event_iter = None
        thought_delta_buffer: list[str] = []
        if history is not None:
            try:
                llm_logger.info("HISTORY %s", json.dumps(history)[:HISTORY_LOG_MAX])
            except Exception:
                llm_logger.info("HISTORY %s", str(history)[:HISTORY_LOG_MAX])
            if isinstance(history, list) and history and not isinstance(history[0], dict):
                safe_history = history
            else:
                sanitized = _inject_system_prompt(_sanitize_history(history), runner)
                safe_history = _convert_to_model_messages(sanitized)
            try:
                llm_logger.info("HISTORY_SANITIZED %s", json.dumps(safe_history)[:HISTORY_LOG_MAX])
            except Exception:
                llm_logger.info("HISTORY_SANITIZED %s", str(safe_history)[:HISTORY_LOG_MAX])
            event_iter = runner.run_stream_events(prompt_text, message_history=safe_history)
        if event_iter is None:
            event_iter = runner.run_stream_events(prompt_text)
        if asyncio.iscoroutine(event_iter):
            event_iter = await event_iter
        async for event in event_iter:
            if cancel_event.is_set():
                closer = getattr(event_iter, "aclose", None)
                if callable(closer):
                    with contextlib.suppress(Exception):
                        await closer()
                return None, usage

            handled = False
            if on_event is not None:
                if asyncio.iscoroutinefunction(on_event):
                    handled = bool(await on_event(event))
                else:
                    maybe = on_event(event)
                    if inspect.isawaitable(maybe):
                        handled = bool(await maybe)
                    else:
                        handled = bool(maybe)

            if handled:
                continue

            if isinstance(event, str):
                output_parts.append(event)
                stream_logger.info("LLM text chunk=%s", event[:200].replace("\n", "\\n"))
                llm_logger.info("RECV text chunk\n%s", event)
                await on_text(event)
                continue
            if isinstance(event, PartDeltaEvent):
                if isinstance(event.delta, ThinkingPartDelta):
                    delta = getattr(event.delta, "content_delta", None)
                    if delta:
                        stream_logger.info("LLM thinking delta=%s", str(delta)[:200].replace("\n", "\\n"))
                        llm_logger.info("RECV thinking delta\n%s", delta)
                        thought_delta_buffer.append(str(delta))
                elif isinstance(event.delta, TextPartDelta):
                    delta = getattr(event.delta, "content_delta", "")
                    if delta:
                        output_parts.append(delta)
                        stream_logger.info("LLM delta=%s", delta[:200].replace("\n", "\\n"))
                        llm_logger.info("RECV delta\n%s", delta)
                        await on_text(delta)
            elif isinstance(event, PartEndEvent):
                kind = getattr(event.part, "part_kind", "")
                if kind and kind not in {"text", "thinking"}:
                    logging.getLogger("acp_server").debug(
                        "Received PartEndEvent kind=%s content=%s",
                        kind,
                        getattr(event.part, "content", None),
                    )
                part = getattr(event.part, "content", "")
                if kind == "thinking" and on_thought is not None:
                    final_thought = str(part) if part else "".join(thought_delta_buffer)
                    thought_delta_buffer.clear()
                    if final_thought:
                        stream_logger.info("LLM thinking=%s", final_thought[:200].replace("\n", "\\n"))
                        llm_logger.info("RECV thinking\n%s", final_thought)
                        await on_thought(final_thought)
                elif part and not output_parts:
                    output_parts.append(part)
                    stream_logger.info("LLM part_end=%s", str(part)[:200].replace("\n", "\\n"))
                    llm_logger.info("RECV part_end\n%s", part)
                    await on_text(part)
            elif isinstance(event, AgentRunResultEvent):
                result = getattr(event, "result", None)
                if result is not None and getattr(result, "output", None):
                    full = getattr(result, "output")
                    usage = getattr(result, "usage", None)
                    stream_logger.info("LLM final output=%s", str(full)[:200].replace("\n", "\\n"))
                    llm_logger.info("RECV final output\n%s", full)
                    try:
                        getter = getattr(result, "all_messages", None)
                        if callable(getter):
                            msgs = getter()
                            llm_logger.info("MODEL_MESSAGES %s", str(msgs)[:HISTORY_LOG_MAX])
                            if store_messages is not None:
                                try:
                                    store_messages(msgs)
                                except Exception:
                                    pass
                    except Exception:
                        pass
                    if not output_parts:
                        try:
                            await on_text(str(full))
                        except Exception:
                            pass
                    return str(full), usage
    except (ai_exc.ModelRetry, ai_exc.UnexpectedModelBehavior) as exc:
        msg = str(exc)
        stream_logger.warning("LLM validation failure: %s", msg)
        llm_logger.warning("LLM validation failure: %s", msg)
        friendly = "Model output failed validation; please retry or adjust the request."
        return friendly, usage
    except Exception as exc:  # pragma: no cover - provider errors
        msg = str(exc)
        stream_logger.warning("LLM stream error: %s", msg)
        llm_logger.warning("LLM stream error: %s", msg)
        return f"Provider error: {msg}", None

    if thought_delta_buffer and on_thought is not None:
        final_thought = "".join(thought_delta_buffer)
        if final_thought:
            with contextlib.suppress(Exception):
                await on_thought(final_thought)

    return ("".join(output_parts) or None), usage
