"""An ACP fixture built on the SDK alone, with no Isaac server imports."""

import asyncio
import sys
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
        self.cancelled = asyncio.Event()

    async def cancel(self, session_id: str, **_: object) -> None:
        self.cancelled.set()

    async def initialize(self, protocol_version: int, **_: object) -> schema.InitializeResponse:
        return schema.InitializeResponse(
            protocol_version=protocol_version,
            agent_info=schema.Implementation(name="sdk-demo", title="SDK Demo", version="1.0"),
            agent_capabilities=schema.AgentCapabilities(
                load_session=True,
                session_capabilities=schema.SessionCapabilities(list=schema.SessionListCapabilities())
                if "--list-sessions" in sys.argv
                else None,
            ),
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
        if session_id not in {"demo", "remote-new", "remote-old"}:
            raise acp.RequestError.invalid_params({"sessionId": session_id})
        await self.client.session_update(
            session_id, acp.update_agent_message(acp.text_block("Session history replayed over ACP."))
        )
        return schema.LoadSessionResponse(config_options=options(self.model, self.verbose))

    async def list_sessions(self, cwd: str, cursor: str | None = None, **_: object) -> schema.ListSessionsResponse:
        if "--list-sessions" not in sys.argv:
            raise acp.RequestError.method_not_found("session/list")
        return schema.ListSessionsResponse(
            sessions=[
                schema.SessionInfo(
                    session_id="remote-new" if cursor else "remote-old",
                    cwd=cwd,
                    title="New session" if cursor else "Old session",
                    updated_at="2026-01-02T00:00:00Z" if cursor else "2026-01-01T00:00:00Z",
                )
            ],
            next_cursor=None if cursor else "next",
        )

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
        first = prompt[0]
        if isinstance(first, schema.TextContentBlock) and first.text in {"Wait for cancel", "Wait then end"}:
            await self.client.session_update(session_id, acp.update_agent_message(acp.text_block("Turn started")))
            if first.text == "Wait for cancel":
                await self.cancelled.wait()
                return schema.PromptResponse(stop_reason="cancelled")
            await asyncio.sleep(0.15)
            await self.client.session_update(session_id, acp.update_agent_message(acp.text_block("Turn finished")))
            return schema.PromptResponse(stop_reason="end_turn")
        if isinstance(first, schema.TextContentBlock) and first.text.startswith("Echo "):
            await self.client.session_update(session_id, acp.update_agent_message(first))
            return schema.PromptResponse(stop_reason="end_turn")
        if isinstance(first, schema.TextContentBlock) and first.text == "Fail the turn":
            raise acp.RequestError.internal_error({"details": "Set the provider credentials and try again."})
        if isinstance(first, schema.TextContentBlock) and first.text == "Ask a form question":
            answer = await self.client.create_elicitation(
                "Which runtime should the example use?",
                schema.ElicitationFormSessionMode(
                    session_id=session_id,
                    requested_schema=schema.ElicitationSchema(
                        type="object",
                        properties={
                            "runtime": schema.ElicitationStringPropertySchema(
                                type="string", title="Runtime", enum=["Python", "JavaScript"]
                            ),
                            "note": schema.ElicitationStringPropertySchema(type="string", title="Other details"),
                        },
                        required=["runtime"],
                    ),
                ),
            )
            reply = str(answer.content) if isinstance(answer, schema.AcceptElicitationResponse) else answer.action
            await self.client.session_update(session_id, acp.update_agent_message(acp.text_block(f"Answer: {reply}")))
            return schema.PromptResponse(stop_reason="end_turn")
        if isinstance(first, schema.TextContentBlock) and first.text == "Show tool activity":
            for number in range(2):
                tool_id = f"read-{number}"
                await self.client.session_update(
                    session_id,
                    schema.ToolCallStart(
                        session_update="tool_call", tool_call_id=tool_id, title="Read file", kind="read"
                    ),
                )
                await self.client.session_update(
                    session_id,
                    schema.ToolCallProgress(
                        session_update="tool_call_update",
                        tool_call_id=tool_id,
                        status="completed",
                        content=[acp.tool_content(acp.text_block("Routine output"))],
                    ),
                )
            await self.client.session_update(
                session_id,
                schema.ToolCallProgress(
                    session_update="tool_call_update",
                    tool_call_id="failure",
                    title="Run tests",
                    status="failed",
                    content=[acp.tool_content(acp.text_block("Test failed with details"))],
                ),
            )
            return schema.PromptResponse(stop_reason="end_turn")
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
