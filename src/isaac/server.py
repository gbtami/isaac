"""Compose Harness capabilities and expose them through its upstream ACP adapter."""

from collections.abc import Callable

import acp
from acp import schema
from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models import Model
from pydantic_ai.workspaces import LocalWorkspaceBackend
from pydantic_ai_harness.code_mode import CodeMode
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.compaction import ReportContextUsage
from pydantic_ai_harness.experimental.acp import ACPSession, ACPSessionConfig, PydanticAIACPAgent
from pydantic_ai_harness.permissions import Permissions
from pydantic_ai_harness.planning import Planning, SqlitePlanStore
from pydantic_ai_harness.skills import Skills

from isaac import __version__
from isaac.auth import model_resolver
from isaac.compaction import AutomaticCompaction
from isaac.config import SessionSettings, Settings, session_database
from isaac.mcp import mcp_capabilities
from isaac.session_permissions import code_mode_approvals
from isaac.session_store import SqliteSessionStore
from isaac.session_titles import SessionTitles
from isaac.session_ui import SessionUI


def build_agent(settings: Settings, *, model: Model | str | None = None) -> Agent[SessionSettings, str]:
    """Bind capabilities at construction so Harness self-delegates inherit them."""
    capabilities: list[AbstractCapability[SessionSettings]] = [
        Coder(),
        Permissions(lambda ctx: ctx.deps.permissions),
        AutomaticCompaction(settings.context_window),
    ]
    if settings.skill_dirs:
        capabilities.append(Skills(settings.skill_dirs, missing_directories="skip"))
    if settings.code_mode:
        capabilities.append(CodeMode())
    # The adapter supplies the selected model at run time. Startup and session setup need no API key.
    return Agent(model, name="isaac", deps_type=SessionSettings, capabilities=capabilities, defer_model_check=True)


def build_acp_adapter(
    settings: Settings, agent: Agent[SessionSettings, str], *, resolve_model: Callable[[str], Model | str]
) -> PydanticAIACPAgent[SessionSettings, str]:
    """Share the upstream adapter's model selection with the slash commands."""
    database = session_database()
    store = SqliteSessionStore(database)

    async def configure_session(session: ACPSession) -> ACPSessionConfig[SessionSettings]:
        plan = SqlitePlanStore(str(database), session=session.session_id)
        ui = SessionUI(session, plan, store, adapter)
        titles = SessionTitles(session, store)
        capabilities: list[AbstractCapability[SessionSettings]] = [
            Planning(store=plan, tools=["write_plan"]),
            ReportContextUsage(context_window=settings.context_window),
            ui,
            titles,
            *mcp_capabilities(session),
        ]
        if settings.code_mode:
            capabilities.append(code_mode_approvals(session))
        await ui.advertise()
        await ui.publish_plan(force=True)
        await titles.restore()
        return ACPSessionConfig(
            deps=SessionSettings(permissions=settings.permissions),
            workspace=LocalWorkspaceBackend(session.cwd),
            capabilities=capabilities,
        )

    adapter = PydanticAIACPAgent(
        agent,
        name="isaac",
        version=__version__,
        models=settings.models,
        model_resolver=resolve_model,
        session_config=configure_session,
        session_store=store,
        mcp_capabilities=schema.McpCapabilities(http=True, sse=True),
    )
    return adapter


async def serve(settings: Settings) -> None:
    """Let Harness own the complete ACP lifecycle and the agent's tool lifecycle."""
    agent = build_agent(settings)
    adapter = build_acp_adapter(settings, agent, resolve_model=model_resolver())
    async with agent:
        await acp.run_agent(adapter, use_unstable_protocol=True)
