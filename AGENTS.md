# Isaac

Isaac is an ACP coding server built on Pydantic AI Harness. It has no bundled client.
The old custom agent/client architecture has been removed.

## Runtime

- `uv run isaac` and `uv run python -m isaac` start the same ACP stdio server.
- Configure `ISAAC_MODEL` in the environment or the shared platform config `.env`, or pass `--model`.
- `uv run isaac login` signs in to ChatGPT/Codex using core's browser OAuth flow, outside ACP.
- Use Toad as a client: `cd ~/toad && uv run toad acp "isaac" --project-dir ~/playground`.

## Design

- Use Harness `Coder` and public capabilities for coding behavior.
- Use `pydantic_ai_harness.experimental.acp.run_acp_stdio` for the complete ACP layer.
- Keep model selection on the adapter's ACP Session Config Options path.
- Reuse core's Codex OAuth flow/provider and credential-source protocol; keep browser login outside ACP.
- Bind coding capabilities at agent construction so self-delegation inherits them.
- Keep planning and its ACP presentation session-local through the session factory.
- Use public capability hooks for the small plan/slash-command bridge; do not replace the adapter.
- Persist conversations through Harness's `SessionStore` protocol and plans through its `SqlitePlanStore`.
- Keep save/load/replay on the upstream ACP adapter's path; Toad discovers sessions through its own history.
- Never modify Toad or other client repositories; accommodate their limitations within Isaac.
- Configure each session through `ACPSessionConfig`, with `LocalWorkspaceBackend(session.cwd)`.
- Attach ACP-provided MCP toolsets through core `Toolset` capabilities.
- Keep stdout exclusive to ACP JSON-RPC; diagnostics go to stderr.
- Do not reintroduce custom tools, ACP handlers, prompt runners, model catalogs, or a client UI.
- Do not depend on or copy CLAI2-specific code. Its Harness composition can be studied in
  `~/pydantic-ai/src/pydantic_clai2`; upstream ACP documentation is in
  `~/pydantic-ai/docs/harness/acp.md`.
- Harness ACP is experimental. Keep its release series pinned and follow its SDK dependency range.

## Modules

- `src/isaac/cli.py`: command line and process startup.
- `src/isaac/config.py`: shared environment and settings.
- `src/isaac/auth.py`: core Codex browser-login bridge and private, atomic credential storage.
- `src/isaac/server.py`: Harness composition and ACP launch.
- `src/isaac/mcp.py`: ACP MCP transport configuration.
- `src/isaac/session_ui.py`: session-local plan projection and slash commands over Harness operations.
- `src/isaac/session_store.py`: SQLite snapshots for the upstream ACP adapter.
- `src/isaac/session_titles.py`: first-prompt title generation and standard ACP title notifications.

## Checks

Use `uv` for dependency and environment management:

```bash
uv sync --locked
uv run ruff format .
uv run ruff check .
uv run mypy src tests
uv run pytest
uv build --wheel
```

For local Toad testing, build a fresh wheel and install it with
`uv tool install --force --reinstall dist/isaac_acp-0.4.0-py3-none-any.whl`.
