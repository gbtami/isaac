# Architecture

Isaac is a stdio ACP server. Its runtime is Pydantic AI Harness, with no bundled client.

```text
Toad / any ACP client
    -> Harness PydanticAIACPAgent over ACP stdio
        -> Pydantic AI Agent
            -> Harness Coder + Permissions + Skills + compaction
            -> per-session Harness Planning + SessionUI
            -> per-session LocalWorkspaceBackend
            -> per-session MCP toolset capabilities
```

## Modules

- `isaac.cli`: argparse, shared environment loading, stderr logging, and event-loop startup.
- `isaac.config`: immutable settings and the platform config `.env` path.
- `isaac.auth`: browser login and atomic credential storage for core's Codex OAuth provider.
- `isaac.compaction`: provider-aware automatic compaction with Harness fallback.
- `isaac.server`: binds Harness capabilities and serves the agent with the public ACP adapter.
- `isaac.mcp`: converts ACP MCP descriptors into public Pydantic AI MCP toolset capabilities.
- `isaac.session_ui`: advertises slash commands and publishes Harness plan and context usage updates over ACP.
- `isaac.session_permissions`: forwards Harness approvals from inside `run_code` to the ACP client.
- `isaac.session_store`: implements Harness's public `SessionStore` protocol using SQLite snapshots.
- `isaac.session_titles`: generates a stable first-prompt title with a bounded, tool-free model request.

Coding capabilities are bound at agent construction so Coder's self-delegation inherits them. The shared
agent has no fixed workspace. Harness calls `configure_session` with each client's `cwd`, and the
returned workspace backend is passed to every run in that session. There is no process-wide `chdir`.
MCP tools are also created separately for each session. MCP commands start in that session's directory.
Each session owns an upstream `SqlitePlanStore` shared by its `Planning` and `SessionUI` capabilities.
After tool results, the UI capability publishes changed full-plan snapshots, including reordered or empty plans.
The main session owns this plan; delegate runs do not overwrite its sidebar.
Plans are keyed by ACP session ID in the same database as conversation snapshots. The session factory
restores the sidebar when a client reloads a session, even though presentation updates are sent outside
the adapter's recorded transcript.

The upstream adapter owns protocol negotiation, sessions, history, cancellation, text/thinking/tool
updates, permission requests, and the `model` session config option. Isaac does not subclass the adapter
or implement protocol endpoints. Isaac keeps the public adapter instance and serves it with `acp.run_agent`,
the same stdio runner used by `run_acp_stdio`. The session capability handles slash commands through the
public `wrap_model_request` hook. `/model` uses the adapter's `set_config_option` path and publishes a standard
config update. `/permissions` edits the live session's dependencies, which Harness reads per tool call and
passes to self-delegates. It starts from the CLI mode on open or resume. Plain `/compact`
tries core's `Model.compact_messages()` and stores the native response at the compaction boundary. Compatible
models resume from that boundary. The preceding messages remain available to a model that can't replay it.
Focused, unsupported, or failed native compaction uses Harness `compact_now`. The Codex subscription
endpoint lacks the standalone compaction route, so it uses Harness directly. The adapter commits all history
updates through its normal turn lifecycle. Automatic compaction applies core's `OpenAICompaction` or
`AnthropicCompaction` settings on the supported first-party endpoints, based on the selected request model.
The inline trigger is 85% of the resolved context window. Harness handles other endpoints, unsupported
inline settings, and estimated windows above 95% where the provider may reject the request before it can
compact. Anthropic's signed on-demand blocks and inline threshold edits cannot be sent together.
Harness `ReportContextUsage` measures each session request after automatic compaction. The session
capability forwards readings with a known context window as standard ACP `usage_update` notifications.

With Code Mode enabled, the session factory attaches Core's `HandleDeferredToolCalls` capability.
Harness's `ask_permissions` resolves nested `run_code` approvals using the client's
`session/request_permission` method. Each approval applies to one operation. The adapter handles
direct approvals and turn cancellation.
Local Harness file and shell tools operate on the agent host;
client-native filesystem/terminal tools are not attached alongside Coder's tools.

The adapter's defaults determine advertised capabilities. MCP HTTP/SSE support is enabled because
the session factory connects both transports. Text prompts are supported by default; Isaac does not
claim image/audio support across its selectable models. Passing `SqliteSessionStore` enables the adapter's
`loadSession` capability, autosave, and `session/load` replay without custom protocol endpoints.

## Persistence

The adapter decides when to save and what history, transcript, and selected model to include. Isaac
serializes the upstream `StoredSession` value with Pydantic and atomically replaces its SQLite row.
Each operation uses a short-lived connection on a worker thread. The database lives in the platform
state directory, or `ISAAC_STATE_DIR`, with private file permissions. Corrupt snapshots raise on load;
they are not silently replaced by empty sessions. The adapter reports save/load errors according to
its own policy.

Toad records session IDs when `loadSession` is advertised, and its history picker invokes `session/load`.
The server does not add its own discovery UI or implement unsupported `session/list` / `session/resume`
methods. Completed conversation turns and plans persist; in-flight runs, shell jobs, and remembered
permission decisions are not restored.

Session titles use a separate SQLite table so updating a conversation snapshot never replaces its name.
A session-local capability captures the first coding prompt and names it after a successful coding run,
using the selected model independently of the foreground history. Naming uses a single request with
an eight-second timeout, with a shortened-prompt fallback. Auxiliary usage is included in the turn's
usage. A resumed session republishes its saved title using ACP `session_info_update`. Clients decide
whether to display it; Isaac does not access client databases or modify client implementations.

## Boundaries

`isaac login` runs independently of ACP. Core's `OpenAICodexOAuthFlow` owns PKCE, state
validation, the localhost callback, and token exchange. Isaac opens the browser and persists
the resulting credentials as a private, atomically replaced JSON file in its config directory.
The adapter's public `model_resolver` hook supplies an `OpenAICodexModel` backed by a shared
`OpenAICodexProvider` for Codex selections, leaving all other model IDs to core. Provider creation
is lazy, so initialization and model selection require no credentials. Core handles expiry,
refresh, and saving rotated tokens through the public `OpenAICodexCredentialSource` protocol.
Sharing the provider also shares core's refresh lock across sessions and models within a server;
separate processes retain core's best-effort coordination through storage. Without an Isaac
credential file, the provider can use core's read-only Codex CLI credential fallback.

All coding behavior is provided by Harness capabilities. Do not copy CLAI2's UI, plugins, model
resolvers, OAuth clients, or session infrastructure. Add a Harness capability when new agent behavior
is needed. Keep Isaac-specific code limited to configuration, launch, and transport integration.

The `pydantic-ai-slim` distribution installs the framework and provider integrations without the
Pydantic AI chat CLI. Transitive libraries may use Rich internally; Isaac has no Rich interface.
