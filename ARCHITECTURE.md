# Architecture

Isaac is a stdio ACP server. Its runtime is Pydantic AI Harness, with no bundled client.

```text
Toad / any ACP client
    -> Harness run_acp_stdio
        -> Pydantic AI Agent
            -> Harness Coder + Permissions + Skills + compaction
            -> per-session Harness Planning + SessionUI
            -> per-session LocalWorkspaceBackend
            -> per-session MCP toolset capabilities
```

## Modules

- `isaac.cli`: argparse, shared environment loading, stderr logging, and event-loop startup.
- `isaac.config`: immutable settings and the platform config `.env` path.
- `isaac.server`: binds Harness capabilities and serves the agent with the public ACP adapter.
- `isaac.mcp`: converts ACP MCP descriptors into public Pydantic AI MCP toolset capabilities.
- `isaac.session_ui`: advertises slash commands and projects Harness plans through standard ACP updates.

Coding capabilities are bound at agent construction so Coder's self-delegation inherits them. The shared
agent has no fixed workspace. Harness calls `configure_session` with each client's `cwd`, and the
returned workspace backend is passed to every run in that session. There is no process-wide `chdir`.
MCP tools are also created separately for each session. MCP commands start in that session's directory.
Each session owns an upstream `InMemoryPlanStore` shared by its `Planning` and `SessionUI` capabilities.
After tool results, the UI capability publishes changed full-plan snapshots, including reordered or empty plans.
The main session owns this plan; delegate runs do not overwrite its sidebar.

The upstream adapter owns protocol negotiation, sessions, history, cancellation, text/thinking/tool
updates, permission requests, and the `model` session config option. Isaac does not subclass the adapter
or implement protocol endpoints. The session capability handles `/help`, `/plan`, and `/compact` through
the public `wrap_model_request` hook, returning normal streamed/history-recorded responses. Manual compaction
uses Harness `compact_now`; history updates still commit through the adapter's normal turn lifecycle.
Local Harness file and shell tools operate on the agent host;
client-native filesystem/terminal tools are not attached alongside Coder's tools.

The adapter's defaults determine advertised capabilities. MCP HTTP/SSE support is enabled because
the session factory connects both transports. Text prompts are supported by default; Isaac does not
claim image/audio support across its selectable models. Persistence is absent.

## Boundaries

All coding behavior is provided by Harness capabilities. Do not copy CLAI2's UI, plugins, model
resolvers, OAuth clients, or session infrastructure. Add a Harness capability when new agent behavior
is needed. Keep Isaac-specific code limited to configuration, launch, and transport integration.

The `pydantic-ai-slim` distribution installs the framework and provider integrations without the
Pydantic AI chat CLI. Transitive libraries may use Rich internally; Isaac has no Rich interface.
