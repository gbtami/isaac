"""Present SDK callbacks and retain the metadata for the active ACP session."""

import asyncio
from collections import Counter
from dataclasses import dataclass, field
from time import monotonic

from acp import schema
from acp.client.connection import ClientSideConnection

from isaac_client.display import Terminal, visible
from isaac_client.elicitation import answer_form
from isaac_client.history import SessionHistory

ConfigOption = schema.SessionConfigOptionSelect | schema.SessionConfigOptionBoolean
ToolContent = schema.ContentToolCallContent | schema.FileEditToolCallContent | schema.TerminalToolCallContent


@dataclass
class SessionState:
    session_id: str | None = None
    title: str | None = None
    options: list[ConfigOption] = field(default_factory=list)
    modes: schema.SessionModeState | None = None
    commands: list[schema.AvailableCommand] = field(default_factory=list)
    usage: schema.UsageUpdate | None = None
    plan: list[schema.PlanEntry] = field(default_factory=list)
    loading: bool = False


class TerminalClient:
    def __init__(self, terminal: Terminal) -> None:
        self.terminal = terminal
        self.history: SessionHistory | None = None
        self.session = SessionState()
        self.agent = "ACP agent"
        self._tools: dict[str, tuple[str, str | None]] = {}
        self._routine_tools: Counter[str] = Counter()
        self._turn_started: float | None = None
        self._speaking: str | None = None
        self.cancelling = False
        self._permission_menus: set[asyncio.Task[int | None]] = set()
        self._question_menus: set[asyncio.Task[schema.CreateElicitationResponse]] = set()

    def remember(self) -> None:
        if self.history is not None and self.session.session_id is not None and not self.session.loading:
            self.history.remember(self.session.session_id, self.session.title)

    def toolbar(self) -> str:
        fields = []
        if self._turn_started is not None:
            if self._permission_menus:
                status = "Waiting for approval"
            elif self._question_menus:
                status = "Waiting for answers"
            elif self.cancelling:
                status = "Cancelling"
            elif self._tools:
                status = next(reversed(self._tools.values()))[0]
            else:
                status = "Working"
            fields.append(f"{status} ({int(monotonic() - self._turn_started)}s)")
        session = self.session
        if session.plan:
            completed = sum(item.status == "completed" for item in session.plan)
            fields.append(f"Plan {completed}/{len(session.plan)}")
        if session.usage:
            fields.append(f"Context {session.usage.used:,}/{session.usage.size:,}")
        fields.append(session.title or self.agent)
        for option in session.options:
            value = next(
                (item.name for item in self.choices(option) if item.value == option.current_value),
                str(option.current_value),
            )
            fields.append(f"{option.name} {value}")
        if session.modes and not any(self.is_option(option, "mode") for option in session.options):
            fields.append(f"Mode {session.modes.current_mode_id}")
        return visible("  ·  ".join(fields)).replace("\n", " ")

    async def request_permission(
        self,
        session_id: str,
        tool_call: schema.ToolCallUpdate,
        options: list[schema.PermissionOption],
        **_: object,
    ) -> schema.RequestPermissionResponse:
        if session_id == self.session.session_id and not self.cancelling:
            self.finish_output()
            stored = self._tools.get(tool_call.tool_call_id)
            title = tool_call.title or (stored[0] if stored else tool_call.tool_call_id)
            self.terminal.write(f"Approve tool call: {title}", style="bold yellow")
            self._show_content(tool_call.content or [])
            menu = asyncio.create_task(
                self.terminal.choose("Permission", [option.name for option in options], toolbar=self.toolbar)
            )
            self._permission_menus.add(menu)
            try:
                choice = await menu
            except asyncio.CancelledError:
                if not self.cancelling:
                    raise
                choice = None
            finally:
                self._permission_menus.discard(menu)
            if choice is not None and not self.cancelling:
                return schema.RequestPermissionResponse(
                    outcome=schema.AllowedOutcome(outcome="selected", option_id=options[choice].option_id)
                )
        return schema.RequestPermissionResponse(outcome=schema.DeniedOutcome(outcome="cancelled"))

    async def create_elicitation(
        self, message: str, mode: schema.ElicitationMode, **_: object
    ) -> schema.CreateElicitationResponse:
        cancelled = schema.CancelElicitationResponse(action="cancel")
        if self.cancelling or not self.terminal.interactive:
            return cancelled
        if isinstance(mode, schema.ElicitationFormSessionMode) and mode.session_id != self.session.session_id:
            return cancelled
        if not isinstance(mode, schema.ElicitationFormSessionMode | schema.ElicitationFormRequestMode):
            return schema.DeclineElicitationResponse(action="decline")
        self.finish_output()
        menu = asyncio.create_task(answer_form(self.terminal, self.agent, message, mode.requested_schema, self.toolbar))
        self._question_menus.add(menu)
        try:
            response = await menu
        except asyncio.CancelledError:
            if not self.cancelling:
                raise
            return cancelled
        finally:
            self._question_menus.discard(menu)
        return cancelled if self.cancelling else response

    def cancel_interactions(self) -> None:
        self.cancelling = True
        for menu in (*self._permission_menus, *self._question_menus):
            menu.cancel()

    async def session_update(self, session_id: str, update: object, **_: object) -> None:
        session = self.session
        if session.session_id is not None and session_id != session.session_id:
            return
        if isinstance(update, schema.SessionInfoUpdate):
            if update.title:
                session.title = " ".join(visible(update.title).split())[:160] or session.title
                self.remember()
        elif isinstance(update, schema.UsageUpdate):
            session.usage = update
        elif isinstance(update, schema.ConfigOptionUpdate):
            session.options = update.config_options
        elif isinstance(update, schema.CurrentModeUpdate):
            if session.modes:
                session.modes.current_mode_id = update.current_mode_id
        elif isinstance(update, schema.AvailableCommandsUpdate):
            session.commands = update.available_commands
        elif isinstance(update, (schema.UserMessageChunk, schema.AgentMessageChunk, schema.AgentThoughtChunk)):
            if not isinstance(update.content, schema.TextContentBlock):
                return
            speaker = (
                "You"
                if isinstance(update, schema.UserMessageChunk)
                else "Thinking"
                if isinstance(update, schema.AgentThoughtChunk)
                else self.agent
            )
            if self._speaking != speaker:
                self.finish_output()
                self.terminal.write(f"{speaker}: ", end="", style="bold")
                self._speaking = speaker
            self.terminal.write(
                update.content.text, end="", style="dim" if isinstance(update, schema.AgentThoughtChunk) else None
            )
        elif isinstance(update, schema.ToolCallStart):
            self._tools[update.tool_call_id] = (update.title, update.kind)
            if update.content and update.kind not in {"read", "search"}:
                self.finish_output()
                self.terminal.write(f"Tool: {update.title}", style="cyan")
                self._show_content(update.content)
        elif isinstance(update, schema.ToolCallProgress):
            self._update_tool(update)
        elif isinstance(update, schema.AgentPlanUpdate):
            session.plan = update.entries

    def _update_tool(self, update: schema.ToolCallProgress) -> None:
        title, kind = self._tools.get(update.tool_call_id, (update.tool_call_id, None))
        title, kind = update.title or title, update.kind or kind
        content = update.content or []
        if update.status in {"completed", "failed"}:
            self._tools.pop(update.tool_call_id, None)
            if (
                update.status == "completed"
                and kind in {"read", "search"}
                and not any(isinstance(block, schema.FileEditToolCallContent) for block in content)
            ):
                self._routine_tools[kind] += 1
                return
            self.finish_output()
            self.terminal.write(
                f"{update.status.title()}: {title}", style="red" if update.status == "failed" else "dim"
            )
        else:
            self._tools[update.tool_call_id] = (title, kind)
            if kind in {"read", "search"}:
                return
        if content:
            self.finish_output()
            self._show_content(content)

    def start_turn(self) -> None:
        self.finish_output()
        self._tools.clear()
        self._turn_started = monotonic()

    def end_turn(self) -> None:
        self.finish_output()
        self._tools.clear()
        self._turn_started = None

    def finish_output(self) -> None:
        self.finish_message()
        if self._routine_tools:
            summary = ", ".join(f"{name} {count}" for name, count in self._routine_tools.items())
            self.terminal.write(f"Completed: {summary}", style="dim")
            self._routine_tools.clear()

    def finish_message(self) -> None:
        if self._speaking is not None:
            self.terminal.write("")
            self._speaking = None

    def _show_content(self, content: list[ToolContent]) -> None:
        for block in content:
            if isinstance(block, schema.FileEditToolCallContent):
                self.terminal.diff(block)
            elif isinstance(block, schema.ContentToolCallContent) and isinstance(
                block.content, schema.TextContentBlock
            ):
                self.terminal.write(block.content.text)

    @staticmethod
    def is_option(option: ConfigOption, category: str) -> bool:
        return option.category == category or option.id == category

    @staticmethod
    def choices(option: ConfigOption) -> list[schema.SessionConfigSelectOption]:
        if isinstance(option, schema.SessionConfigOptionBoolean):
            return []
        return [
            choice
            for item in option.options
            for choice in (item.options if isinstance(item, schema.SessionConfigSelectGroup) else [item])
        ]

    async def select_config(
        self, connection: ClientSideConnection, session_id: str, category: str | None = None
    ) -> None:
        options = [option for option in self.session.options if category is None or self.is_option(option, category)]
        if not options:
            self.terminal.write("The agent offered no matching session settings.")
            return
        index = (
            0
            if len(options) == 1
            else await self.terminal.choose("Setting", [option.name for option in options], search=True)
        )
        if index is None:
            return
        option = options[index]
        choices = self.choices(option)
        values: list[str | bool] = [choice.value for choice in choices]
        labels = [choice.name for choice in choices]
        if isinstance(option, schema.SessionConfigOptionBoolean):
            values = [True, False]
            labels = ["On", "Off"]
        labels = [
            f"{label} (current)" if value == option.current_value else label
            for label, value in zip(labels, values, strict=True)
        ]
        selected = await self.terminal.choose(option.name, labels, search=True)
        if selected is not None:
            response = await connection.set_config_option(
                session_id=session_id, config_id=option.id, value=values[selected]
            )
            self.session.options = response.config_options
            self.terminal.write(f"{option.name}: {labels[selected]}")
