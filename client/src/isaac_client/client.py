"""Present SDK callbacks and retain the metadata for the active ACP session."""

from acp import schema
from acp.client.connection import ClientSideConnection

from isaac_client.display import Terminal, visible
from isaac_client.history import SessionHistory

ConfigOption = schema.SessionConfigOptionSelect | schema.SessionConfigOptionBoolean
ToolContent = schema.ContentToolCallContent | schema.FileEditToolCallContent | schema.TerminalToolCallContent


class TerminalClient:
    def __init__(self, terminal: Terminal) -> None:
        self.terminal = terminal
        self.history: SessionHistory | None = None
        self.session_id: str | None = None
        self.agent = "ACP agent"
        self.title: str | None = None
        self.options: list[ConfigOption] = []
        self.modes: schema.SessionModeState | None = None
        self.commands: list[schema.AvailableCommand] = []
        self.usage: schema.UsageUpdate | None = None
        self.plan: list[schema.PlanEntry] = []
        self._tools: dict[str, str] = {}
        self._speaking: str | None = None

    def remember(self) -> None:
        if self.history is not None and self.session_id is not None:
            self.history.remember(self.session_id, self.title)

    def toolbar(self) -> str:
        fields = [self.title or self.agent]
        for option in self.options:
            value = next(
                (item.name for item in self.choices(option) if item.value == option.current_value),
                str(option.current_value),
            )
            fields.append(f"{option.name} {value}")
        if self.modes and not any(self.is_option(option, "mode") for option in self.options):
            fields.append(f"Mode {self.modes.current_mode_id}")
        if self.usage:
            fields.append(f"Context {self.usage.used:,}/{self.usage.size:,}")
        return visible("  ·  ".join(fields)).replace("\n", " ")

    async def request_permission(
        self,
        session_id: str,
        tool_call: schema.ToolCallUpdate,
        options: list[schema.PermissionOption],
        **_: object,
    ) -> schema.RequestPermissionResponse:
        if session_id == self.session_id:
            self.finish_message()
            title = tool_call.title or self._tools.get(tool_call.tool_call_id) or tool_call.tool_call_id
            self.terminal.write(f"Approve tool call: {title}", style="bold yellow")
            self._show_content(tool_call.content or [])
            choice = await self.terminal.choose("Permission", [option.name for option in options])
            if choice is not None:
                return schema.RequestPermissionResponse(
                    outcome=schema.AllowedOutcome(outcome="selected", option_id=options[choice].option_id)
                )
        return schema.RequestPermissionResponse(outcome=schema.DeniedOutcome(outcome="cancelled"))

    async def session_update(self, session_id: str, update: object, **_: object) -> None:
        if self.session_id is not None and session_id != self.session_id:
            return
        if isinstance(update, schema.SessionInfoUpdate):
            if update.title:
                self.title = " ".join(visible(update.title).split())[:160] or self.title
                self.remember()
        elif isinstance(update, schema.UsageUpdate):
            self.usage = update
        elif isinstance(update, schema.ConfigOptionUpdate):
            self.options = update.config_options
        elif isinstance(update, schema.CurrentModeUpdate):
            if self.modes:
                self.modes.current_mode_id = update.current_mode_id
        elif isinstance(update, schema.AvailableCommandsUpdate):
            self.commands = update.available_commands
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
                self.finish_message()
                self.terminal.write(f"{speaker}: ", end="", style="bold")
                self._speaking = speaker
            self.terminal.write(
                update.content.text, end="", style="dim" if isinstance(update, schema.AgentThoughtChunk) else None
            )
        elif isinstance(update, schema.ToolCallStart):
            self.finish_message()
            self._tools[update.tool_call_id] = update.title
            self.terminal.write(f"Tool: {update.title}", style="cyan")
            self._show_content(update.content or [])
        elif isinstance(update, schema.ToolCallProgress):
            if update.title:
                self._tools[update.tool_call_id] = update.title
            if update.status in {"completed", "failed"}:
                self.finish_message()
                title = self._tools.pop(update.tool_call_id, update.tool_call_id)
                self.terminal.write(f"{update.status.title()}: {title}", style="dim")
            if update.content:
                self.finish_message()
                self._show_content(update.content)
        elif isinstance(update, schema.AgentPlanUpdate):
            self.finish_message()
            self.plan = update.entries
            self.terminal.plan(self.plan)

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
        options = [option for option in self.options if category is None or self.is_option(option, category)]
        if not options:
            self.terminal.write("The agent offered no matching session settings.")
            return
        index = 0 if len(options) == 1 else await self.terminal.choose("Setting", [option.name for option in options])
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
        selected = await self.terminal.choose(option.name, labels)
        if selected is not None:
            response = await connection.set_config_option(
                session_id=session_id, config_id=option.id, value=values[selected]
            )
            self.options = response.config_options
            self.terminal.write(f"{option.name}: {labels[selected]}")
