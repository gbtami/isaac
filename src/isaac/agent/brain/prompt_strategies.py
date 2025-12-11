"""Prompt strategy helpers for planning/execution flows."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict

from acp.contrib.tool_calls import ToolCallTracker
from acp.helpers import (
    plan_entry,
    session_notification,
    text_block,
    tool_content,
    update_agent_message,
    update_plan,
)
from acp.schema import SessionNotification
from pydantic_ai.messages import FunctionToolCallEvent, FunctionToolResultEvent, RetryPromptPart

from isaac.agent.brain.planner import parse_plan_from_text
from isaac.agent.brain.prompt import EXECUTOR_PROMPT
from isaac.agent.runner import stream_with_runner
from isaac.agent.tools.run_command import (
    RunCommandContext,
    reset_run_command_context,
    set_run_command_context,
)
from isaac.agent.usage import normalize_usage


@dataclass
class PromptContext:
    """Shared context for prompt strategy handlers."""

    session_id: str
    prompt_text: str
    history: Any
    cancel_event: asyncio.Event
    runner: Any
    planner: Any
    single_runner: Any | None
    strategy_has_planning: bool
    store_model_messages: Callable[[Any], None]


@dataclass
class PromptStrategyDef:
    """Prompt strategy metadata and handler reference."""

    strategy_id: str
    label: str
    description: str
    has_planning: bool
    handler: Callable[["PromptStrategyManager", PromptContext], Awaitable[Any]]


@dataclass
class StrategyEnv:
    """Environment/state shared by strategy handlers."""

    session_modes: Dict[str, str]
    session_last_chunk: Dict[str, str | None]
    session_strategies: Dict[str, str]
    delegate_tool_enabled: set[str]
    default_strategy: str
    send_update: Callable[[SessionNotification], Awaitable[None]]
    request_run_permission: Callable[[str, str, str, str | None], Awaitable[bool]]
    refresh_history: Callable[[str], Any]
    set_usage: Callable[[str, Any | None], None]


class PromptStrategyManager:
    """Manage available prompt strategies and dispatch prompt handling."""

    def __init__(self, env: StrategyEnv) -> None:
        self.env = env
        self._strategies: Dict[str, PromptStrategyDef] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        self._strategies = {
            "handoff": PromptStrategyDef(
                strategy_id="handoff",
                label="Programmatic hand-off",
                description="Two-phase planning then execution (planner model then executor).",
                has_planning=True,
                handler=self._run_programmatic_strategy,
            ),
            "delegation": PromptStrategyDef(
                strategy_id="delegation",
                label="Agent delegation",
                description="Executor delegates planning via delegate_plan tool before executing.",
                has_planning=True,
                handler=self._run_delegation_strategy,
            ),
            "single": PromptStrategyDef(
                strategy_id="single",
                label="Single agent workflow",
                description="One agent handles everything; does not emit plan updates.",
                has_planning=False,
                handler=self._run_single_agent_strategy,
            ),
            "plan_only": PromptStrategyDef(
                strategy_id="plan_only",
                label="Planning only",
                description="Only produce a plan; no execution phase.",
                has_planning=True,
                handler=self._run_plan_only_strategy,
            ),
        }

    def current_strategy_id(self, session_id: str) -> str:
        return self.env.session_strategies.get(session_id, self.env.default_strategy)

    def describe(self, session_id: str) -> str:
        current = self.current_strategy_id(session_id)
        lines = [f"Current prompt strategy: {current}"]
        lines.append("Available strategies:")
        for strategy_id, entry in self._strategies.items():
            prefix = "*" if strategy_id == current else "-"
            planning = " (planning updates)" if entry.has_planning else " (no plan updates)"
            lines.append(f"{prefix} {strategy_id} - {entry.label}: {entry.description}{planning}")
        return "\n".join(lines)

    def set_strategy(self, strategy_id: str, session_id: str) -> str:
        normalized = (strategy_id or "").strip().lower()
        if not normalized:
            return "Usage: /strategy <id> (use /strategies to list options)"
        if normalized not in self._strategies:
            return f"Unknown strategy '{strategy_id}'. Use /strategies to list available options."
        self.env.session_strategies[session_id] = normalized
        self.env.delegate_tool_enabled.discard(session_id)
        return f"Prompt strategy set to {normalized}."

    def is_valid_strategy(self, strategy_id: str) -> bool:
        return strategy_id in self._strategies

    async def run(
        self,
        session_id: str,
        prompt_text: str,
        *,
        history: Any,
        cancel_event: asyncio.Event,
        runner: Any,
        planner: Any,
        single_runner: Any | None,
        store_model_messages: Callable[[Any], None],
    ) -> Any:
        strategy_id = self.current_strategy_id(session_id)
        strategy = self._strategies.get(strategy_id) or self._strategies[self.env.default_strategy]
        ctx = PromptContext(
            session_id=session_id,
            prompt_text=prompt_text,
            history=history,
            cancel_event=cancel_event,
            runner=runner,
            planner=planner,
            single_runner=single_runner,
            strategy_has_planning=strategy.has_planning,
            store_model_messages=store_model_messages,
        )
        return await strategy.handler(ctx)

    # --- Strategy handlers ---

    async def _run_programmatic_strategy(self, ctx: PromptContext) -> Any:
        plan_update, plan_text, plan_usage = await self._run_planning_phase(ctx)
        history = ctx.history
        if plan_update:
            await self.env.send_update(session_notification(ctx.session_id, plan_update))

        executor_prompt = self._prepare_executor_prompt(
            ctx.prompt_text, plan_update=plan_update, plan_response=plan_text
        )
        return await self._run_execution_phase(
            ctx.session_id,
            ctx.runner,
            executor_prompt,
            history=history,
            cancel_event=ctx.cancel_event,
            store_model_messages=ctx.store_model_messages,
            plan_update=plan_update,
            plan_response=plan_text,
            plan_usage=plan_usage,
            allow_plan_parse=ctx.strategy_has_planning,
        )

    async def _run_delegation_strategy(self, ctx: PromptContext) -> Any:
        self._ensure_delegate_tool(ctx.session_id, ctx.runner, ctx.planner)
        delegate_hint = (
            "Use the delegate_plan tool first to request a 3-6 step plan, then execute it."
        )
        executor_prompt = f"{delegate_hint}\n\n{ctx.prompt_text}"
        plan_seen = {"sent": False}

        async def _plan_hook(tool_name: str, raw_output: dict[str, Any]) -> None:
            if tool_name != "delegate_plan":
                return
            plan_update = self._plan_update_from_raw(raw_output)
            if plan_update:
                plan_seen["sent"] = True
                await self.env.send_update(session_notification(ctx.session_id, plan_update))

        response = await self._run_execution_phase(
            ctx.session_id,
            ctx.runner,
            executor_prompt,
            history=ctx.history,
            cancel_event=ctx.cancel_event,
            store_model_messages=ctx.store_model_messages,
            plan_result_hook=_plan_hook,
            allow_plan_parse=ctx.strategy_has_planning,
        )
        if getattr(response, "stop_reason", "") == "end_turn" and not plan_seen["sent"]:
            maybe_plan = self._plan_update_from_raw(
                {"content": self.env.session_last_chunk.get(ctx.session_id)}
            )
            if maybe_plan:
                await self.env.send_update(session_notification(ctx.session_id, maybe_plan))
        return response

    async def _run_single_agent_strategy(self, ctx: PromptContext) -> Any:
        single_runner = ctx.single_runner or ctx.runner
        single_hint = "If the task is non-trivial, share a short plan before executing the steps."
        executor_prompt = f"{ctx.prompt_text}\n\n{single_hint}"
        return await self._run_execution_phase(
            ctx.session_id,
            single_runner,
            executor_prompt,
            history=ctx.history,
            cancel_event=ctx.cancel_event,
            store_model_messages=ctx.store_model_messages,
            allow_plan_parse=ctx.strategy_has_planning,
        )

    async def _run_plan_only_strategy(self, ctx: PromptContext) -> Any:
        plan_update, _, plan_usage = await self._run_planning_phase(ctx)
        if plan_update:
            await self.env.send_update(session_notification(ctx.session_id, plan_update))
        self.env.set_usage(ctx.session_id, normalize_usage(plan_usage))
        return self._prompt_end()

    # --- Shared helpers ---

    async def _run_planning_phase(
        self, ctx: PromptContext
    ) -> tuple[Any | None, str | None, Any | None]:
        plan_chunks: list[str] = []

        async def _plan_chunk(chunk: str) -> None:
            plan_chunks.append(chunk)

        plan_response, plan_usage = await stream_with_runner(
            ctx.planner,
            ctx.prompt_text,
            _plan_chunk,
            ctx.cancel_event,
            history=ctx.history,
            store_messages=ctx.store_model_messages,
        )
        combined_plan_text = plan_response or "".join(plan_chunks)
        if combined_plan_text.startswith("Provider error:"):
            msg = combined_plan_text.removeprefix("Provider error:").strip()
            await self.env.send_update(
                session_notification(
                    ctx.session_id,
                    update_agent_message(
                        text_block(f"Model/provider error during planning: {msg}")
                    ),
                )
            )
            # Discard provider error text so it isn't reused as a pseudo-plan.
            combined_plan_text = ""
        elif combined_plan_text.lower().startswith("Model output failed validation"):
            await self.env.send_update(
                session_notification(
                    ctx.session_id,
                    update_agent_message(text_block(combined_plan_text)),
                )
            )
            # Drop validation failure text to avoid feeding it into execution prompts.
            combined_plan_text = ""

        plan_update = parse_plan_from_text(combined_plan_text or "")
        if not plan_update and combined_plan_text:
            entries = [plan_entry(combined_plan_text)]
            plan_update = update_plan(entries)
        return plan_update, combined_plan_text or plan_response, plan_usage

    def _prepare_executor_prompt(
        self,
        prompt_text: str,
        *,
        plan_update: Any | None = None,
        plan_response: str | None = None,
    ) -> str:
        plan_lines = [getattr(e, "content", "") for e in getattr(plan_update, "entries", []) or []]
        if plan_lines:
            plan_block = "\n".join(f"- {line}" for line in plan_lines if line)
            return f"{prompt_text}\n\nPlan:\n{plan_block}\n\n{EXECUTOR_PROMPT}"
        if plan_response:
            return f"{prompt_text}\n\nPlan:\n{plan_response}\n\n{EXECUTOR_PROMPT}"
        return prompt_text

    def _make_chunk_sender(self, session_id: str) -> Callable[[str], Awaitable[None]]:
        async def _push_chunk(chunk: str) -> None:
            last = self.env.session_last_chunk.get(session_id)
            if chunk == last:
                return
            self.env.session_last_chunk[session_id] = chunk
            await self.env.send_update(
                session_notification(
                    session_id,
                    update_agent_message(text_block(chunk)),
                )
            )

        return _push_chunk

    def _build_runner_event_handler(
        self,
        session_id: str,
        tool_trackers: Dict[str, ToolCallTracker],
        run_command_ctx_tokens: Dict[str, Any],
        plan_result_hook: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None,
    ) -> Callable[[Any], Awaitable[bool]]:
        async def _handle_runner_event(event: Any) -> bool:
            if isinstance(event, FunctionToolCallEvent):
                tool_name = getattr(event.part, "tool_name", None) or ""
                raw_args = getattr(event.part, "args", None)
                args = raw_args if isinstance(raw_args, dict) else {}
                tracker = ToolCallTracker(id_factory=lambda: event.tool_call_id)
                tool_trackers[event.tool_call_id] = tracker
                start = tracker.start(
                    external_id=event.tool_call_id,
                    title=tool_name,
                    status="in_progress",
                    raw_input={"tool": tool_name, **(args if isinstance(args, dict) else {})},
                )
                await self.env.send_update(session_notification(session_id, start))
                if tool_name == "run_command":
                    allowed = True
                    mode = self.env.session_modes.get(session_id, "ask")
                    if mode == "ask":
                        allowed = await self.env.request_run_permission(
                            session_id=session_id,
                            tool_call_id=event.tool_call_id,
                            command=str(args.get("command") if isinstance(args, dict) else "")
                            if args
                            else "",
                            cwd=args.get("cwd") if isinstance(args, dict) else None,
                        )
                    if not allowed:
                        denied = tracker.progress(
                            external_id=event.tool_call_id,
                            status="failed",
                            raw_output={
                                "tool": tool_name,
                                "content": None,
                                "error": "permission denied",
                            },
                            content=[
                                tool_content(text_block("Command blocked: permission denied"))
                            ],
                        )
                        await self.env.send_update(session_notification(session_id, denied))
                        return True

                    token = set_run_command_context(
                        RunCommandContext(request_permission=lambda *_: True)
                    )
                    run_command_ctx_tokens[event.tool_call_id] = token
                return True

            if isinstance(event, FunctionToolResultEvent):
                token = run_command_ctx_tokens.pop(event.tool_call_id, None)
                if token is not None:
                    reset_run_command_context(token)
                tracker = tool_trackers.pop(event.tool_call_id, None) or ToolCallTracker(
                    id_factory=lambda: event.tool_call_id
                )
                result_part = event.result
                tool_name = getattr(result_part, "tool_name", None) or ""
                content = getattr(result_part, "content", None)
                raw_output: dict[str, Any] = {}
                if isinstance(content, dict):
                    raw_output.update(content)
                else:
                    raw_output["content"] = content
                raw_output.setdefault("tool", tool_name)
                status = "completed"
                if isinstance(result_part, RetryPromptPart):
                    raw_output["error"] = result_part.model_response()
                    status = "failed"
                else:
                    raw_output.setdefault("error", None)
                summary = raw_output.get("error") or raw_output.get("content") or ""
                if plan_result_hook is not None:
                    try:
                        await plan_result_hook(tool_name, raw_output)
                    except Exception:
                        pass
                progress = tracker.progress(
                    external_id=event.tool_call_id,
                    status=status,
                    raw_output=raw_output,
                    content=[tool_content(text_block(str(summary)))] if summary else None,
                )
                await self.env.send_update(session_notification(session_id, progress))
                return True

            return False

        return _handle_runner_event

    async def _run_execution_phase(
        self,
        session_id: str,
        runner: Any,
        executor_prompt: str,
        *,
        history: Any,
        cancel_event: asyncio.Event,
        store_model_messages: Callable[[Any], None],
        plan_update: Any | None = None,
        plan_response: str | None = None,
        plan_usage: Any | None = None,
        plan_result_hook: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None,
        allow_plan_parse: bool = True,
    ) -> Any:
        tool_trackers: Dict[str, ToolCallTracker] = {}
        run_command_ctx_tokens: Dict[str, Any] = {}
        _push_chunk = self._make_chunk_sender(session_id)
        handler = self._build_runner_event_handler(
            session_id, tool_trackers, run_command_ctx_tokens, plan_result_hook
        )

        response_text, usage = await stream_with_runner(
            runner,
            executor_prompt,
            _push_chunk,
            cancel_event,
            history=history,
            on_event=handler,
            store_messages=store_model_messages,
        )
        if response_text is None:
            return self._prompt_cancel()
        if response_text.startswith("Provider error:"):
            msg = response_text.removeprefix("Provider error:").strip()
            await self.env.send_update(
                session_notification(
                    session_id,
                    update_agent_message(text_block(f"Model/provider error: {msg}")),
                )
            )
            return self._prompt_end()

        exec_plan_update = None
        if allow_plan_parse and not plan_update and not plan_response and plan_result_hook is None:
            exec_plan_update = parse_plan_from_text(response_text or "")
        if exec_plan_update:
            await self.env.send_update(session_notification(session_id, exec_plan_update))
        if response_text == "":
            await _push_chunk(response_text)
        combined_usage = normalize_usage(usage) or normalize_usage(plan_usage)
        self.env.set_usage(session_id, combined_usage)
        return self._prompt_end()

    def _ensure_delegate_tool(self, session_id: str, runner: Any, planner: Any) -> None:
        if session_id in self.env.delegate_tool_enabled:
            return
        if runner is None or planner is None:
            return

        @runner.tool(name="delegate_plan")  # type: ignore[misc]
        async def delegate_plan(task: str) -> dict[str, Any]:
            try:
                result = await planner.run(task)
                data = getattr(result, "data", None)
                steps = getattr(data, "steps", None) if data is not None else None
                plan_steps = list(steps or [])
                if not plan_steps and isinstance(data, str):
                    plan_steps = [data]
                if not plan_steps and hasattr(result, "output") and isinstance(result.output, str):
                    plan_steps = [result.output]
                plan_text = "\n".join(str(step) for step in plan_steps if step) or task
                return {"plan": plan_steps, "content": plan_text, "tool": "delegate_plan"}
            except Exception as exc:  # noqa: BLE001
                return {"plan": [], "content": None, "error": str(exc), "tool": "delegate_plan"}

        self.env.delegate_tool_enabled.add(session_id)

    def _plan_update_from_raw(self, raw_output: dict[str, Any]) -> Any | None:
        plan = raw_output.get("plan")
        content = raw_output.get("content") or ""
        candidates: list[str] = []
        if isinstance(plan, list):
            candidates.extend(str(item) for item in plan if item)
        if not candidates and isinstance(plan, str):
            candidates.append(plan)
        if not candidates and content:
            parsed = parse_plan_from_text(str(content))
            if parsed:
                return parsed
            candidates.append(str(content))
        if not candidates:
            return None
        entries = [plan_entry(item) for item in candidates if item]
        if not entries:
            return None
        return update_plan(entries)

    @staticmethod
    def _prompt_cancel() -> Any:
        from acp.schema import PromptResponse  # local import to avoid cycles

        return PromptResponse(stop_reason="cancelled")

    @staticmethod
    def _prompt_end() -> Any:
        from acp.schema import PromptResponse  # local import to avoid cycles

        return PromptResponse(stop_reason="end_turn")
