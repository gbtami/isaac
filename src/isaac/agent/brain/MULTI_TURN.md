Multi-Turn Prompting in Isaac
=============================

This document explains how the Isaac agent maintains context across multiple
prompts while staying within model limits and ACP constraints. It focuses on
the agent-side behavior in `src/isaac/agent/brain/`, with protocol formatting
handled by the ACP adapter.

Key Goals
---------

- Preserve relevant state across turns (user intent, tool outcomes, file paths).
- Avoid invalid tool-call history that some providers reject.
- Prevent context blow-up with token-aware compaction.
- Keep delegate sub-agents isolated by default.

Where History Lives
-------------------

1) In-memory session state
   - `PromptHandler` keeps `SessionState.history`, a list of `{role, content}`
     dictionaries that represent user and assistant messages plus tool summaries.
   - `SessionState.recent_files` tracks the last few edited file paths.

2) ACP replay history
   - ACP updates are stored on disk via `SessionStore` as `history.jsonl`.
   - ACP history is used for session replay, not for model prompt assembly.

3) Delegate sessions
   - Delegate tools (planner/review/coding) keep their own short summaries when
     `carryover=True`. They never inherit the full parent history.

Prompt Assembly
---------------

- The model prompt history comes from `SessionState.history`.
- Before sending to the model, Isaac injects a short system hint listing recent
  files touched (most recent last). This resolves ambiguous follow-ups such as
  "split it" by anchoring on the last edited file.
- History is sanitized before passing to pydantic-ai to avoid empty or malformed
  messages.

Tool Call Context and History
-----------------------------

- The stream event handler records tool results and produces a tool summary that
  is appended to `SessionState.history`.
- Summaries include concrete data:
  - `run_command`: command, cwd, and truncated stdout/stderr.
  - `edit_file`/`apply_patch`: path + diff or summary.
  - delegate tools: task + summary.
- This keeps the model aware of what changed without forcing it to re-read files.

Tool Call Normalization
-----------------------

- Providers can reject truncated tool histories. Isaac avoids sending tool call
  metadata in history by default, and only keeps user/assistant content.
- Tool call summaries remain in history as plain text, which is safe and helps
  multi-turn continuity.

Token-Aware Compaction
----------------------

Compaction is triggered by token usage rather than raw message count.

- The model context limit is read from `models.json` (or models.dev) using
  `models.get_context_limit()`.
- A compaction threshold is computed as `context_limit * 0.7`.
- Isaac estimates history tokens and also uses provider `RunUsage` totals when
  available; the larger value is used to decide if compaction is needed.
- When compaction triggers, Isaac rebuilds history to mirror Codex behavior:
  it keeps recent user messages (bounded by a token budget) and appends a
  summary message with a fixed summary prefix. Assistant/tool messages are not
  kept directly; their effects must appear in the summary.

This mimics Codex-style behavior where compaction is based on token budget and
retains a stable summary format.

Codex-Style Summary Prompt and Prefix
-------------------------------------

Isaac uses the same compaction prompt and summary prefix style as Codex:

- The compaction prompt asks for a concise handoff summary with progress,
  constraints, and next steps.
- The summary message is stored as a user message prefixed with a fixed
  summary header. Future compactions can detect and exclude this summary.

This ensures the summary is explicit and machine-detectable while staying
compatible with normal prompt assembly.

Compaction Input Trimming and Fallback
--------------------------------------

Compaction is only useful if the summary model sees a clean, bounded input.
Isaac reduces compaction inputs and guards against bad summaries:

- Before summarizing, Isaac builds a compaction-only history that trims large
  tool outputs (diffs, stdout/stderr) and caps each message length.
- If the trimmed history is still too large, Isaac drops the oldest messages
  until the compaction prompt fits within the model context budget.
- If the summary response is empty or clearly unusable (for example: "no earlier
  conversation"), Isaac replaces it with a deterministic fallback summary.
- The fallback summary extracts the last user request, recent file updates,
  commands run, delegate usage, and the last assistant response.

This keeps multi-turn continuity even when the summary model returns a failure.
It also reduces token usage for the compaction step itself by avoiding huge
tool outputs.

When compaction runs, Isaac emits a standard notification via the ACP adapter
so clients know that earlier context was summarized.

Delegate Agent Isolation
------------------------

Delegate tools run in isolated sessions by default:

- They receive only the explicit task and optional context.
- When carryover is enabled, only a short summary from the same delegate session
  is included (no full history).

This keeps delegate roles focused and avoids cross-contamination of context.

Important Files
---------------

- `src/isaac/agent/brain/prompt_handler.py`
  - Session history, recent files, compaction, prompt assembly.
- `src/isaac/agent/brain/prompt_runner.py`
  - Tool progress handling and tool summary generation.
- `src/isaac/agent/brain/compaction.py`
  - Compaction prompt, summary prefix, token estimates, and history rebuild.
- `src/isaac/agent/brain/history_utils.py`
  - History trimming and usage token extraction helpers.
- `src/isaac/agent/brain/plan_helpers.py`
  - Parsing planner tool output into PlanSteps.
- `src/isaac/agent/brain/recent_files.py`
  - Recent-file tracking and context injection helpers.
- `src/isaac/agent/brain/session_state.py`
  - SessionState dataclass for in-memory prompt state.
- `src/isaac/agent/brain/session_ops.py`
  - Session lifecycle helpers (model load, runner build, error response).
- `src/isaac/agent/brain/tool_events.py`
  - Tool history summaries and tool kind classification.
- `src/isaac/agent/acp/prompt_env.py`
  - ACP adapter for message, tool, and plan updates.
- `src/isaac/agent/acp/plan_updates.py`
  - Converts PlanSteps into ACP plan updates.
- `src/isaac/agent/acp/history.py`
  - ACP update replay into model-friendly history.
- `src/isaac/agent/runner.py`
  - History sanitization before sending to pydantic-ai.
- `src/isaac/agent/subagents/delegate_tools.py`
  - Delegate isolation and carryover summaries.

Comparison to Codex
-------------------

Codex maintains a structured transcript of conversation items (user messages,
assistant messages, tool calls, tool outputs) and builds each prompt directly
from those items. It also normalizes call/output pairs and compacts history
based on token usage rather than message count.

Isaac mirrors the same ideas in a lightweight way:

- Structured context: tool outputs are recorded in history as detailed summaries
  (paths, diffs, stdout/stderr) instead of only high-level notes.
- Token-aware compaction: the compaction trigger uses the model context limit
  and provider usage totals when available.
- Isolation: delegate tools only see explicit task context and a short carryover
  summary, preserving the "fresh context" behavior used by specialized Codex
  sub-tasks.

Isaac does not persist full prompt-ready transcripts the way Codex rollouts do.
Instead it stores summaries in memory and ACP updates for replay. If you need
Codex-style persistent transcripts later, the natural extension is to store
structured tool outputs in the session history log and replay them into model
history on resume.
If you update tool outputs or add new tools, ensure their summaries include
critical state (paths, commands, outcomes) so follow-up prompts remain grounded.
