"""Forward Harness permission checks inside Code Mode to the ACP client."""

from pathlib import Path

import acp
from acp import schema
from pydantic_ai import DeferredToolRequests, DeferredToolResults, RunContext
from pydantic_ai.capabilities import HandleDeferredToolCalls
from pydantic_ai_harness.experimental.acp import ACPSession, ToolCallPresentation, default_coding_presenter
from pydantic_ai_harness.permissions import PermissionDecision, PermissionRequest, ask_permissions


def code_mode_approvals(session: ACPSession) -> HandleDeferredToolCalls[None]:
    """Handle only nested Harness approvals. The adapter owns direct tool approvals."""

    async def approve(request: PermissionRequest) -> PermissionDecision:
        presentation = default_coding_presenter(request.call) or ToolCallPresentation()
        content = [
            item.model_copy(update={"path": str(Path(session.cwd, item.path))})
            if isinstance(item, schema.FileEditToolCallContent)
            else item
            for item in presentation.content
        ]
        action = "Write" if request.action == "write" else "Run"
        response = await session.client.request_permission(
            session_id=session.session_id,
            tool_call=schema.ToolCallUpdate(
                tool_call_id=request.call.tool_call_id,
                title=f"{action} {request.subject}",
                kind=presentation.kind or ("edit" if request.action == "write" else "execute"),
                content=content or [acp.tool_content(acp.text_block(request.reason))],
                locations=[
                    location.model_copy(update={"path": str(Path(session.cwd, location.path))})
                    for location in presentation.locations
                ]
                or None,
            ),
            options=[
                schema.PermissionOption(kind="allow_once", name="Allow once", option_id="allow_once"),
                schema.PermissionOption(kind="reject_once", name="Reject", option_id="reject_once"),
            ],
        )
        outcome = response.outcome
        # ACP session/cancel cancels the wait itself. A dismissed dialog never approves the call.
        return PermissionDecision(
            approved=isinstance(outcome, schema.AllowedOutcome) and outcome.option_id == "allow_once"
        )

    async def handle(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults | None:
        if ctx.tool_name != "run_code":
            return None
        return await ask_permissions(requests, approve)

    return HandleDeferredToolCalls(handler=handle)
