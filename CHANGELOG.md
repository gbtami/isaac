# Changelog

All notable changes to Isaac are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-07-10

Isaac 0.4.0 is a major internal modernization of the ACP agent and reference
client. It should be treated as an architecture boundary rather than a routine
dependency update. Python 3.10 is no longer supported, and compatibility paths
for the previous experimental internals have intentionally been removed.

### Added

- Structured coding memory for durable file observations, edits, command results,
  plans, findings, risks, tests, and follow-up work.
- Context selection and compaction designed for longer multi-turn coding sessions,
  including restart and session-load continuity.
- Planner, coding, and review delegates with normalized artifact propagation back
  into the parent session.
- ACP-visible plan, tool-call, progress, thinking, and usage updates.
- Session-scoped model and mode selection, interactive client commands, and MCP
  server configuration forwarding.
- Optional Pydantic AI Harness filesystem, shell, and CodeMode experiments behind
  the `harness` extra and explicit environment flags.
- Optimistic SHA-256 preconditions for mutating file tools and bounded pagination
  for large file, search, and command results.

### Changed

- Raised the minimum supported Python version from 3.10 to 3.11 and declared
  support for Python 3.11 through 3.14.
- Migrated to Agent Client Protocol SDK 0.11 and Pydantic AI 2.x.
- Reworked agent construction around composable Pydantic AI capabilities instead
  of post-construction tool and prompt mutation.
- Bound model-facing tools to the active ACP session workspace and approved
  additional directories.
- Restored model-facing session state on ACP `session/load` rather than replaying
  only the client transcript.
- Isolated per-session model selection and cleared session brain state on close.
- Moved tool observation, approval handling, recent-file context, and plan updates
  to the Pydantic AI event-stream/capability boundary.
- Standardized development and release workflows around `uv`.

### Security

- Hardened filesystem tools against workspace escapes, symlink traversal, binary
  misuse, stale writes, and writes to protected paths.
- Added catastrophic shell-command blocking plus optional command allowlist and
  denylist policies.
- Added deterministic command timeouts, process-group cleanup, and separate
  bounded stdout/stderr diagnostics.
- Removed common secret-bearing environment variables from command subprocesses;
  additional variable-name patterns can be configured with
  `ISAAC_COMMAND_ENV_DENYLIST`.
- Added stricter ACP session identifier validation and bounded stdio handling.

### Packaging

- Added the `py.typed` marker so the published package exposes its inline typing
  information to type checkers.
- Added CI coverage for every declared Python version and made Ruff formatting a
  required lint check.

[0.4.0]: https://github.com/gbtami/isaac/compare/v0.3.1...v0.4.0
