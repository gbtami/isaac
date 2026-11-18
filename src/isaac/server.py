"""Compose Harness capabilities and expose them through its upstream ACP adapter."""

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
from pydantic_ai_harness.planning import InMemoryPlanStore, Planning
from pydantic_ai_harness.skills import Skills

from isaac import __version__
from isaac.config import Settings
from isaac.mcp import mcp_capabilities
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


async def configure_session(session: ACPSession) -> ACPSessionConfig[None]:
    """Use the client's workspace without changing this process's working directory."""
    plan = InMemoryPlanStore()
    ui = SessionUI(session, plan)
    mcp = mcp_capabilities(session)
    await ui.advertise()
    return ACPSessionConfig(
        deps=None,
        workspace=LocalWorkspaceBackend(session.cwd),
        capabilities=[Planning(store=plan, tools=["write_plan"]), ui, *mcp],
    )


async def serve(settings: Settings) -> None:
    """Let Harness own the complete ACP lifecycle and the agent's tool lifecycle."""
    agent = build_agent(settings)
    async with agent:
        await run_acp_stdio(
            agent,
            name="isaac",
            version=__version__,
            models=settings.models,
            session_config=configure_session,
            mcp_capabilities=schema.McpCapabilities(http=True, sse=True),
        )
