"""Compose Harness capabilities and expose them through its upstream ACP adapter."""

from functools import partial

from acp import schema
from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models import Model
from pydantic_ai.workspaces import LocalWorkspaceBackend
from pydantic_ai_harness.code_mode import CodeMode
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.compaction import (
    SlidingWindowCompaction,
    SummarizingCompaction,
    TieredCompaction,
)
from pydantic_ai_harness.experimental.acp import ACPSession, ACPSessionConfig, run_acp_stdio
from pydantic_ai_harness.permissions import Permissions
from pydantic_ai_harness.planning import Planning, SqlitePlanStore
from pydantic_ai_harness.skills import Skills

from isaac import __version__
from isaac.auth import model_resolver
from isaac.config import Settings, session_database
from isaac.mcp import mcp_capabilities
from isaac.session_store import SqliteSessionStore
from isaac.session_titles import SessionTitles
from isaac.session_ui import SessionUI


def build_agent(settings: Settings, *, model: Model | str | None = None) -> Agent[None, str]:
    """Bind capabilities at construction so Harness self-delegates inherit them."""
    capabilities: list[AbstractCapability[None]] = [
        Coder(),
        Permissions(settings.permissions),
        TieredCompaction(
            tiers=[SummarizingCompaction(), SlidingWindowCompaction(keep_messages=20)],
            target_fraction=0.85,
            context_window=settings.context_window,
        ),
    ]
    if settings.skill_dirs:
        capabilities.append(Skills(settings.skill_dirs, missing_directories="skip"))
    if settings.code_mode:
        capabilities.append(CodeMode())
    # The adapter supplies the selected model at run time. Startup and session setup need no API key.
    return Agent(model, name="isaac", capabilities=capabilities, defer_model_check=True)


async def configure_session(session: ACPSession, *, store: SqliteSessionStore) -> ACPSessionConfig[None]:
    """Use the client's workspace without changing this process's working directory."""
    plan = SqlitePlanStore(str(session_database()), session=session.session_id)
    ui = SessionUI(session, plan, store)
    titles = SessionTitles(session, store)
    mcp = mcp_capabilities(session)
    await ui.advertise()
    await ui.publish_plan(force=True)
    await titles.restore()
    return ACPSessionConfig(
        deps=None,
        workspace=LocalWorkspaceBackend(session.cwd),
        capabilities=[Planning(store=plan, tools=["write_plan"]), ui, titles, *mcp],
    )


async def serve(settings: Settings) -> None:
    """Let Harness own the complete ACP lifecycle and the agent's tool lifecycle."""
    agent = build_agent(settings)
    store = SqliteSessionStore(session_database())
    async with agent:
        await run_acp_stdio(
            agent,
            name="isaac",
            version=__version__,
            models=settings.models,
            model_resolver=model_resolver(),
            session_config=partial(configure_session, store=store),
            session_store=store,
            mcp_capabilities=schema.McpCapabilities(http=True, sse=True),
        )
