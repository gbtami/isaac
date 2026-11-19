"""An ACP fixture built on the SDK alone, with no Isaac server imports."""

import asyncio
from typing import cast

import acp
from acp import schema


def options(model: str, verbose: bool) -> list[schema.SessionConfigOptionSelect | schema.SessionConfigOptionBoolean]:
    return [
        schema.SessionConfigOptionSelect(
            type="select",
            id="provider-choice",
            category="model",
            name="Available models",
            current_value=model,
            options=[
                schema.SessionConfigSelectGroup(
                    group="test",
                    name="Demo models",
                    options=[
                        schema.SessionConfigSelectOption(value="one", name="Model One"),
                        schema.SessionConfigSelectOption(value="two", name="Model Two"),
                    ],
                )
            ],
        ),
        schema.SessionConfigOptionBoolean(type="boolean", id="verbosity", name="Verbose", current_value=verbose),
    ]


class DemoAgent:
    model = "one"
    verbose = False

    def on_connect(self, conn: acp.Client) -> None:
        self.client = conn

    async def initialize(self, protocol_version: int, **_: object) -> schema.InitializeResponse:
        return schema.InitializeResponse(
            protocol_version=protocol_version,
            agent_info=schema.Implementation(name="sdk-demo", title="SDK Demo", version="1.0"),
            agent_capabilities=schema.AgentCapabilities(load_session=True),
        )

    async def new_session(self, cwd: str, **_: object) -> schema.NewSessionResponse:
        await self.client.session_update(
            session_id="demo",
            update=schema.AvailableCommandsUpdate(
                session_update="available_commands_update",
                available_commands=[schema.AvailableCommand(name="demo", description="Run the demo")],
            ),
        )
        return schema.NewSessionResponse(session_id="demo", config_options=options(self.model, self.verbose))

    async def load_session(self, cwd: str, session_id: str, **_: object) -> schema.LoadSessionResponse:
        if session_id != "demo":
            raise acp.RequestError.invalid_params({"sessionId": session_id})
        await self.client.session_update(
            session_id, acp.update_agent_message(acp.text_block("Session history replayed over ACP."))
        )
        return schema.LoadSessionResponse(config_options=options(self.model, self.verbose))

    async def set_config_option(
        self, config_id: str, session_id: str, value: str | bool, **_: object
    ) -> schema.SetSessionConfigOptionResponse:
        if config_id == "provider-choice" and isinstance(value, str):
            self.model = value
        elif config_id == "verbosity" and isinstance(value, bool):
            self.verbose = value
        else:
            raise acp.RequestError.invalid_params({"configId": config_id})
        return schema.SetSessionConfigOptionResponse(config_options=options(self.model, self.verbose))

    async def prompt(
        self,
        prompt: list[
            schema.TextContentBlock
            | schema.ImageContentBlock
            | schema.AudioContentBlock
            | schema.ResourceContentBlock
            | schema.EmbeddedResourceContentBlock
        ],
        session_id: str,
        **_: object,
    ) -> schema.PromptResponse:
        await self.client.session_update(
            session_id, schema.SessionInfoUpdate(session_update="session_info_update", title="Agent-assigned title")
        )
        await self.client.session_update(
            session_id, schema.UsageUpdate(session_update="usage_update", size=1000, used=25)
        )
        await self.client.session_update(
            session_id,
            schema.AgentPlanUpdate(
                session_update="plan",
                entries=[schema.PlanEntry(content="Check the result", priority="medium", status="in_progress")],
            ),
        )
        decision = await self.client.request_permission(
            session_id,
            schema.ToolCallUpdate(
                tool_call_id="edit",
                title="Edit demo.txt",
                content=[
                    schema.FileEditToolCallContent(type="diff", path="demo.txt", old_text="before", new_text="after")
                ],
            ),
            [
                schema.PermissionOption(option_id="allow", name="Allow once", kind="allow_once"),
                schema.PermissionOption(option_id="reject", name="Reject once", kind="reject_once"),
            ],
        )
        result = decision.outcome.option_id if isinstance(decision.outcome, schema.AllowedOutcome) else "cancelled"
        await self.client.session_update(
            session_id,
            acp.update_agent_message(acp.text_block(f"model={self.model} verbose={self.verbose} choice={result}")),
        )
        return schema.PromptResponse(stop_reason="end_turn")


if __name__ == "__main__":
    asyncio.run(acp.run_agent(cast(acp.Agent, DemoAgent()), use_unstable_protocol=True))
