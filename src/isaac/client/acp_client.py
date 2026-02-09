"""Protocol-facing ACP client implementation and session update handling."""

from __future__ import annotations

from typing import Any

import json
import logging
from acp import (
    Client,
    CreateTerminalRequest,
    CreateTerminalResponse,
    KillTerminalCommandRequest,
    KillTerminalCommandResponse,
    ReleaseTerminalRequest,
    ReleaseTerminalResponse,
    RequestError,
    RequestPermissionResponse,
    SessionNotification,
    TerminalOutputRequest,
    TerminalOutputResponse,
    WaitForTerminalExitRequest,
    WaitForTerminalExitResponse,
)
from acp.schema import (
    AgentMessageChunk,
    AgentThoughtChunk,
    AgentPlanUpdate,
    AllowedOutcome,
    AudioContentBlock,
    AvailableCommandsUpdate,
    ConfigOptionUpdate,
    CurrentModeUpdate,
    DeniedOutcome,
    EmbeddedResourceContentBlock,
    FileEditToolCallContent,
    ImageContentBlock,
    ResourceContentBlock,
    SessionConfigSelectOption,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
)

from isaac.client.client_terminal import ClientTerminalManager
from isaac.client.display import (
    print_agent_text,
    print_diff,
    print_mode_update,
    print_plan,
    print_file_edit_diff,
    print_thought,
    print_tool,
)
from isaac.client.session_state import SessionUIState
from isaac.log_utils import log_context as log_ctx, log_event

logger = logging.getLogger(__name__)
MODE_CONFIG_KEY = "mode"
MODEL_CONFIG_KEY = "model"


def _iter_select_values(options: list[Any]) -> set[str]:
    values: set[str] = set()
    for option in options:
        if isinstance(option, SessionConfigSelectOption):
            values.add(option.value)
            continue
        nested = getattr(option, "options", None)
        if isinstance(nested, list):
            values.update(_iter_select_values(nested))
    return values


def _semantic_config_key(root_option: Any) -> str | None:
    option_id = str(getattr(root_option, "id", "") or "").lower()
    category = str(getattr(root_option, "category", "") or "").lower()
    name = str(getattr(root_option, "name", "") or "").lower()
    values = _iter_select_values(list(getattr(root_option, "options", []) or []))
    if option_id in {MODE_CONFIG_KEY, "agent_mode"}:
        return MODE_CONFIG_KEY
    if option_id in {MODEL_CONFIG_KEY, "model_id"}:
        return MODEL_CONFIG_KEY
    if category == MODE_CONFIG_KEY or "mode" in name:
        return MODE_CONFIG_KEY
    if category == MODEL_CONFIG_KEY or "model" in name:
        return MODEL_CONFIG_KEY
    if {"ask", "yolo"}.intersection(values):
        return MODE_CONFIG_KEY
    return None


def apply_session_config_options(state: SessionUIState, config_options: list[Any]) -> None:
    """Apply advertised ACP session config options to local UI state."""
    changed = False
    option_ids = dict(state.config_option_ids)
    option_values = dict(state.config_option_values)
    for config_option in config_options:
        root = getattr(config_option, "root", config_option)
        if getattr(root, "type", None) != "select":
            continue
        semantic_key = _semantic_config_key(root)
        if semantic_key is None:
            continue
        option_id = str(getattr(root, "id", "") or "")
        if option_id and option_ids.get(semantic_key) != option_id:
            option_ids[semantic_key] = option_id
            changed = True
        values = _iter_select_values(list(getattr(root, "options", []) or []))
        if values and option_values.get(semantic_key) != values:
            option_values[semantic_key] = values
            changed = True
        current_value = str(getattr(root, "current_value", "") or "")
        if semantic_key == MODE_CONFIG_KEY and current_value and state.current_mode != current_value:
            state.current_mode = current_value
            changed = True
        if semantic_key == MODEL_CONFIG_KEY and current_value and state.current_model != current_value:
            state.current_model = current_value
            changed = True
    if changed:
        state.config_option_ids = option_ids
        state.config_option_values = option_values
        state.notify_changed()


class ACPClient(Client):
    """Minimal ACP client for exercising the protocol endpoints."""

    def __init__(self, state: SessionUIState) -> None:
        self._last_prompt = ""
        self._state = state
        self._terminal_manager = ClientTerminalManager()
        self._logger = logger

    async def request_permission(
        self,
        options,
        session_id: str,
        tool_call: Any,
        **_: Any,
    ) -> RequestPermissionResponse:
        """Prompt the user for a permission choice (Prompt Turn permission flow)."""
        if self._state.cancel_requested:
            return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))
        if self._state.thinking_status is not None:
            self._state.thinking_status.stop()
        with log_ctx(session_id=session_id):
            log_event(
                self._logger,
                "client.permission.request",
                tool=getattr(tool_call, "title", "") or getattr(tool_call, "tool_call_id", ""),
                options=[getattr(opt, "option_id", "<id>") for opt in options],
            )
        try:
            for idx, opt in enumerate(options, start=1):
                label = getattr(opt, "label", opt.option_id)
                print(f"{idx}) {label}")
            choice = input("Permission choice (number): ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(options):
                selection = options[int(choice) - 1].option_id
            else:
                selection = options[0].option_id if options else "default"
        except Exception:
            selection = options[0].option_id if options else "default"
        with log_ctx(session_id=session_id):
            log_event(self._logger, "client.permission.response", selection=selection)
        return RequestPermissionResponse(outcome=AllowedOutcome(option_id=selection, outcome="selected"))

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Handle extension methods from the agent (unused in example client)."""
        return {}

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        """Handle extension notifications from the agent (noop for example client)."""
        return None

    def on_connect(self, *_: Any, **__: Any) -> None:
        """No-op connect hook for compatibility with ACP client interface."""
        return None

    async def write_text_file(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        raise RequestError.method_not_found("fs/write_text_file")

    async def read_text_file(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        raise RequestError.method_not_found("fs/read_text_file")

    async def create_terminal(
        self,
        command: str,
        session_id: str,
        args=None,
        cwd=None,
        env=None,
        output_byte_limit=None,
        **_: Any,
    ) -> CreateTerminalResponse:
        """Create a terminal on the client host per Terminals section."""
        req = CreateTerminalRequest(
            command=command,
            session_id=session_id,
            args=args,
            cwd=cwd,
            env=env,
            output_byte_limit=output_byte_limit,
        )
        return await self._terminal_manager.create_terminal(req)

    async def terminal_output(self, session_id: str, terminal_id: str, **_: Any) -> TerminalOutputResponse:
        """Return terminal output to the agent."""
        req = TerminalOutputRequest(session_id=session_id, terminal_id=terminal_id)
        return await self._terminal_manager.terminal_output(req)

    async def release_terminal(self, session_id: str, terminal_id: str, **_: Any) -> ReleaseTerminalResponse:
        req = ReleaseTerminalRequest(session_id=session_id, terminal_id=terminal_id)
        return await self._terminal_manager.release_terminal(req)

    async def wait_for_terminal_exit(self, session_id: str, terminal_id: str, **_: Any) -> WaitForTerminalExitResponse:
        """Block until the requested client terminal exits."""
        req = WaitForTerminalExitRequest(session_id=session_id, terminal_id=terminal_id)
        return await self._terminal_manager.wait_for_terminal_exit(req)

    async def kill_terminal_command(self, session_id: str, terminal_id: str, **_: Any) -> KillTerminalCommandResponse:
        req = KillTerminalCommandRequest(session_id=session_id, terminal_id=terminal_id)
        return await self._terminal_manager.kill_terminal(req)

    async def kill_terminal(self, *args: Any, **kwargs: Any) -> KillTerminalCommandResponse:
        """Alias for kill_terminal_command to satisfy ACP interface expectations."""
        return await self.kill_terminal_command(*args, **kwargs)

    async def session_update(self, session_id: str, update: SessionNotification | Any, **_: Any) -> None:
        if self._state.thinking_status is not None:
            self._state.thinking_status.stop()
        update_obj = update if not isinstance(update, SessionNotification) else update.update
        update = update_obj or update
        if isinstance(update, CurrentModeUpdate):
            self._state.current_mode = update.current_mode_id
            self._state.notify_changed()
            print_mode_update(self._state.current_mode)
            return
        if isinstance(update, ConfigOptionUpdate):
            apply_session_config_options(self._state, list(update.config_options or []))
            return
        if isinstance(update, AvailableCommandsUpdate):
            local_slashes = self._state.local_slash_commands
            cmds = {}
            for cmd in update.available_commands or []:
                name = f"/{cmd.name}"
                if name in local_slashes:
                    continue
                desc = cmd.description or ""
                hint = ""
                try:
                    hint = getattr(cmd.input.root, "hint", "") if cmd.input else ""
                except Exception:
                    hint = ""
                if hint and not desc:
                    desc = hint
                cmds[name] = desc or hint or ""
            self._state.available_agent_commands = cmds
            return
        if isinstance(update, ToolCallStart):
            title = getattr(update, "title", "")
            kind = getattr(update, "kind", None)
            raw_in = getattr(update, "raw_input", {}) or {}
            cmd = None
            if raw_in.get("tool") == "run_command":
                cmd = raw_in.get("command")
                if isinstance(cmd, dict):
                    cmd = cmd.get("command") or cmd.get("cmd") or cmd
                if isinstance(cmd, str):
                    stripped = cmd.strip()
                    if stripped.startswith("{") and stripped.endswith("}"):
                        try:
                            parsed = json.loads(stripped)
                        except json.JSONDecodeError:
                            parsed = None
                        if isinstance(parsed, dict):
                            cmd = parsed.get("command") or parsed.get("cmd") or cmd
            if cmd is not None and not isinstance(cmd, str):
                cmd = str(cmd)
            message = f"{title}: {cmd}" if cmd else title
            print_tool("start", message, kind=str(kind) if kind else None)
            return
        if isinstance(update, ToolCallProgress):
            raw_out = getattr(update, "raw_output", {}) or {}
            for block in update.content or []:
                if isinstance(block, FileEditToolCallContent) or getattr(block, "type", "") == "diff":
                    try:
                        print_file_edit_diff(
                            getattr(block, "path", "") or "",
                            getattr(block, "old_text", None),
                            getattr(block, "new_text", ""),
                        )
                    except Exception:
                        pass
            if update.status == "completed":
                rc = raw_out.get("returncode")
                err = raw_out.get("error")
                truncated = raw_out.get("truncated")
                summary_bits = []
                if rc is not None:
                    summary_bits.append(f"rc={rc}")
                if err:
                    summary_bits.append(f"error={err}")
                if truncated:
                    summary_bits.append("truncated")
                summary = " ".join(summary_bits) if summary_bits else "done"
                print_tool(update.status, summary, kind=str(getattr(update, "kind", "")) or None)
                delegate_tool = raw_out.get("delegate_tool")
                delegate_output = raw_out.get("content")
                if delegate_tool and delegate_output not in (None, ""):
                    formatted = _format_delegate_output(delegate_tool, delegate_output)
                    if formatted:
                        print_agent_text(formatted)
            else:
                text = raw_out.get("content") or raw_out.get("error") or ""
                print_tool(update.status, text, kind=str(getattr(update, "kind", "")) or None)
                if raw_out.get("delegate_tool"):
                    return
                if getattr(update, "content", None):
                    for item in update.content or []:
                        inner = getattr(item, "content", None)
                        if hasattr(inner, "text") and getattr(inner, "text", None):
                            payload = str(inner.text)
                            if raw_out.get("tool") == "edit_file" and raw_out.get("diff"):
                                print_diff(raw_out["diff"])
                            else:
                                print_agent_text(payload)
            return
        if isinstance(update, AgentPlanUpdate):
            print_plan(update.entries or [])
            return
        if isinstance(update, AgentThoughtChunk):
            if not self._state.show_thinking:
                return
            content = getattr(update, "content", None)
            if content is not None and getattr(content, "text", None):
                print_thought(content.text)
            return
        if not isinstance(update, AgentMessageChunk):
            return

        content = update.content
        display_text: str
        prefix = ""
        if isinstance(content, TextContentBlock):
            display_text = content.text
            if self._state.collect_models:
                self._state.model_buffer = (self._state.model_buffer or []) + [display_text]
            handled_usage, show_usage_text = self._maybe_update_usage_summary(display_text)
            if handled_usage:
                if show_usage_text:
                    if not display_text.endswith("\n"):
                        display_text = f"{display_text}\n"
                    print_agent_text(display_text)
                return
            print_agent_text(display_text)
            return
        if isinstance(content, ImageContentBlock):
            display_text = "<image>"
        elif isinstance(content, AudioContentBlock):
            display_text = "<audio>"
        elif isinstance(content, ResourceContentBlock):
            display_text = content.uri or "<resource>"
        elif isinstance(content, EmbeddedResourceContentBlock):
            display_text = "<resource>"
        else:
            display_text = "<content>"
        rendered = f"{prefix} {display_text}" if prefix else display_text
        print_agent_text(rendered)

    def _maybe_update_usage_summary(self, text: str) -> tuple[bool, bool]:
        """Capture usage lines so the UI can display token/usage summaries."""
        suppress = bool(self._state.suppress_usage_output)
        if text.startswith("Usage:"):
            self._state.usage_summary = text
            self._state.notify_changed()
            return True, not suppress
        if text.startswith("Usage not available") or text.startswith("Usage data unavailable"):
            self._state.usage_summary = text
            self._state.notify_changed()
            return True, not suppress
        return False, False


def _format_delegate_output(tool_name: str, payload: Any) -> str:
    """Render delegate tool payloads into user-facing summaries.

    This keeps delegate results readable while avoiding raw JSON spam for tools
    like the planner that already stream structured updates separately.
    """
    if isinstance(payload, dict):
        if tool_name == "planner" and payload.get("entries"):
            return ""
        summary = payload.get("summary")
        lines: list[str] = []
        if isinstance(summary, str) and summary.strip():
            lines.append(summary.strip())

        def _append_list(title: str, items: Any) -> None:
            if not items:
                return
            if not isinstance(items, list):
                items = [items]
            if not items:
                return
            lines.append(f"{title}:")
            for item in items:
                if isinstance(item, dict):
                    desc = item.get("description") or item.get("summary") or item.get("path") or str(item)
                    severity = item.get("severity")
                    location = item.get("location") or item.get("line")
                    extra = []
                    if severity:
                        extra.append(str(severity))
                    if location:
                        extra.append(str(location))
                    suffix = f" ({', '.join(extra)})" if extra else ""
                    lines.append(f"- {desc}{suffix}")
                else:
                    lines.append(f"- {item}")

        _append_list("Findings", payload.get("findings"))
        _append_list("Files", payload.get("files"))
        _append_list("Tests", payload.get("tests"))
        _append_list("Risks", payload.get("risks"))
        _append_list("Followups", payload.get("followups"))
        if lines:
            return "\n".join(lines).rstrip() + "\n"
        return json.dumps(payload, ensure_ascii=True, indent=2) + "\n"
    if isinstance(payload, list):
        return "\n".join(str(item) for item in payload).rstrip() + "\n"
    return f"{payload}\n"


async def set_mode(
    conn: Any,
    session_id: str,
    state: SessionUIState,
    mode: str,
) -> None:
    await set_session_config_option_value(conn, session_id, state, MODE_CONFIG_KEY, mode)
    state.current_mode = mode
    state.notify_changed()
    print_mode_update(state.current_mode)


async def set_session_config_option_value(
    conn: Any,
    session_id: str,
    state: SessionUIState,
    config_key: str,
    value: str,
) -> None:
    config_id = state.config_option_ids.get(config_key)
    if not config_id:
        raise RuntimeError(f"Agent does not expose a `{config_key}` session config option.")

    setter = getattr(conn, "set_session_config_option", None)
    response: Any
    if callable(setter):
        response = await setter(config_id=config_id, session_id=session_id, value=value)
    else:
        raw_conn = getattr(conn, "_conn", None)
        if raw_conn is None or not hasattr(raw_conn, "send_request"):
            raise RuntimeError("Connection does not support session/set_config_option.")
        response = await raw_conn.send_request(
            "session/set_config_option",
            {"sessionId": session_id, "configId": config_id, "value": value},
        )

    if isinstance(response, dict):
        options = response.get("configOptions") or response.get("config_options") or []
    else:
        options = getattr(response, "config_options", []) or []
    if isinstance(options, list):
        apply_session_config_options(state, options)
