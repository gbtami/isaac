# Isaac Deep Project Review and Next Implementation Plan

## Executive verdict

Isaac is well past a thin ACP compatibility experiment. The project already has a real architecture: an ACP adapter, prompt runner, Pydantic AI capability layer, tool system, session persistence, compaction, recent-file context, and delegated subagents. That is a good foundation.

However, for the next goal — building a strong long-running coding agent — the “brain” is still closer to a promising early implementation than a robust multi-step coding assistant.

Approximate assessment:

| Area | Rating | Verdict |
|---|---:|---|
| ACP protocol shape | 7.5/10 | Mostly solid and modern. |
| Tool safety design | 7/10 | Better than many small agents, but some session-cwd issues are serious. |
| Coding-agent reasoning loop | 6/10 | Good prompt and capability composition, but too dependent on raw recent history. |
| Long multi-step context engineering | 5/10 | Useful compaction exists, but context is lossy and not yet file/task-aware enough. |
| Restart/session continuity | 4/10 | UI history can replay, but brain history is not restored automatically. |

The main conclusion:

> Isaac is good enough for short-to-medium coding sessions, but not yet good enough for reliable multi-hour, multi-step coding work where the agent must remember old reads, design constraints, previous failed attempts, file discoveries, restarted sessions, and delegated work.

The two most important blockers are:

1. ACP `session/load` restores the UI transcript, but does not automatically restore the brain’s working context.
2. Model-triggered tools are not reliably bound to the ACP session workspace/cwd, which can make coding behavior incorrect or surprising.

---

## What is already strong

### 1. The architecture is clean

The separation is generally good:

```text
ACP transport/session layer
        ↓
PromptHandler / PromptRunner / SessionState
        ↓
Pydantic AI agent + capabilities
        ↓
Isaac tools / delegate tools / MCP tools
```

That is the right direction. ACP should remain an adapter, not the brain. Isaac mostly follows that.

The ACP layer, prompt handling layer, and tool execution layer are reasonably separated. This gives the project room to support different clients without rewriting the agent core.

### 2. The Pydantic AI capability usage is a good design choice

The current use of Pydantic AI capabilities is one of the better parts of the design. Isaac already has useful pieces such as:

- system prompt reinjection
- history sanitization
- recent-file transient context
- tool approval policy
- deferred tool-call handling
- event-stream observation

This is a much better foundation than building every behavior manually inside one giant custom loop.

### 3. The ACP surface is reasonably modern

The project exposes session configuration, mode/config options, tool calls, tool progress updates, and assistant updates through the ACP-facing layer.

The design is moving in the correct direction: ACP clients should see Isaac as an agent with proper session lifecycle, tool events, and configurable behavior, while the actual coding intelligence remains behind the ACP boundary.

### 4. Compaction exists, and it is not naive

`brain/compaction.py` is a serious attempt. It has:

- model-aware context limits
- fallback checkpoints
- JSON-ish structured summary target
- overflow retry
- synthetic summary filtering
- model-switch compaction

That is much better than just keeping a growing chat list.

### 5. Delegated subagents are a good strategic direction

The planner/reviewer/coding delegate split is conceptually good. For a coding agent, isolated subtasks are useful because the main agent does not need to pollute the main transcript with every exploratory read/search.

However, delegate integration needs to become more structured and more tightly connected to parent session memory.

---

## Critical issue 1: `session/load` restores the UI, not the brain

This is the biggest correctness problem for ACP compatibility and long sessions.

In `src/isaac/agent/acp/sessions.py`, `load_session()` rebuilds session metadata, initializes the session, loads persisted ACP updates, and replays them to the client.

That is useful for the client UI, but it does not appear to restore the actual `PromptHandler` brain state automatically:

- `SessionState.history`
- recent files
- compaction checkpoint
- selected model state
- durable task context
- previous file observations
- previous tool observations

This creates a dangerous split:

```text
Client sees old conversation.
Brain starts mostly fresh.
Next user prompt assumes the agent remembers.
Agent does not really remember.
```

For a serious coding agent, this is a major issue. The user will naturally continue a loaded session with prompts like:

```text
Apply the fix we discussed earlier.
Continue from where we stopped.
Run the remaining checks.
Use the design decision from before.
```

If Isaac has only replayed UI updates to the client but has not restored the model-facing working context, the agent can answer confidently while missing the actual previous reasoning and observations.

### Recommended fix

On `load_session()`:

1. Load `prompt.json` if it exists.
2. Restore `PromptHandler` state automatically.
3. If no prompt snapshot exists, reconstruct a compact brain history from `history.jsonl`.
4. Restore recent files and compaction checkpoint.
5. Add tests proving that after `session/load`, the next prompt can refer to earlier work.

A useful behavioral test:

```text
turn 1: "Remember that the bug is in src/foo.py and the expected fix is X."
restart/load session
turn 2: "Apply the fix we discussed."
```

The agent should know what “the fix we discussed” means.

---

## Critical issue 2: model tools are not reliably bound to the ACP session cwd

The direct ACP tool-call path seems to know the session cwd. But the model-facing Pydantic tool wrappers do not consistently pass session cwd/additional directories into the tool implementation.

This is serious.

In a coding agent, when the model calls:

```text
read_file("pyproject.toml")
run_command("pytest")
edit_file("src/foo.py", ...)
```

those calls must be relative to the active ACP session workspace, not the server process cwd.

From the code shape, direct ACP execution uses session cwd, but Pydantic model tools go through wrappers in `tools/registration.py` and then `run_tool(...)` without a strong session context/deps object carrying cwd. That makes relative tool calls potentially operate from the wrong directory.

### Why this matters

For a coding agent, this is not a minor edge case. It affects nearly every task:

- reading the wrong file
- running tests in the wrong directory
- editing the wrong path
- failing only under certain launch directories
- behaving differently depending on how the server was started

### Recommended fix

Introduce a real `SessionToolContext` / Pydantic deps object:

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class SessionToolDeps:
    session_id: str
    cwd: Path
    additional_directories: tuple[Path, ...]
    mode: str
    model_id: str
    update_sink: object
```

Then every model-facing tool wrapper should pull from `ctx.deps` and pass cwd explicitly into the underlying tool implementation.

A useful test:

```python
async def test_model_tool_read_file_uses_session_cwd(tmp_path):
    # server process cwd is not tmp_path
    # new ACP session cwd = tmp_path
    # model calls read_file("marker.txt")
    # assert it reads tmp_path / "marker.txt"
```

This should be a P0 fix before relying on Isaac as a serious coding agent.

---

## Critical issue 3: current context engineering is too lossy for long sessions

The current brain mainly relies on:

- `state.history`
- `trim_history(..., 30)`
- synthetic compaction summary
- recent edited files
- selective tool summaries

That is not enough for long coding sessions.

The most problematic part is that every run receives only the last 30 chat messages before the current prompt. This is independent of whether the context window could fit more and independent of whether older messages are highly relevant.

Also, durable summaries appear to exist mainly for:

- `delegate`
- `edit_file`
- `apply_patch`
- `run_command`

But the agent also needs durable observations for:

- `read_file`
- `list_files`
- `code_search`
- `file_summary`
- `fetch_url`

For a coding agent, read/search observations are often more important than edits. The agent’s understanding of the codebase comes mostly from search and read operations.

### Example failure mode

```text
User: "Inspect the architecture and later we will refactor it."
Agent reads 20 files, discovers important constraints.
User asks several follow-ups.
After enough turns, only the last 30 messages remain.
The earlier read/search evidence is gone.
Agent confidently proposes a refactor that violates an old constraint.
```

### Better model

Isaac needs a durable, structured brain transcript, not just chat history.

Suggested event types:

```python
UserMessage
AssistantMessage
ToolCall
ToolResult
FileObservation
FileEdit
CommandResult
PlanUpdate
CompactionCheckpoint
Decision
OpenQuestion
Risk
```

Then build model context from a selector:

```text
system prompt
AGENTS.md
active task summary
current plan
last N user/assistant turns
recent edits
recent command/test results
relevant file observations
relevant previous decisions
compaction checkpoint
current user prompt
```

The context selector should choose relevant history by path, symbol, command, test name, error string, and user-mentioned terms — not just recency.

---

## Critical issue 4: compaction is useful but not sufficient

The compaction code is a good start, but the current shape is too general.

The checkpoint schema is roughly:

```text
progress
key_decisions
user_preferences
remaining_steps
critical_artifacts
risks
```

That is helpful, but for coding you need more specific durable facts:

```text
files_read
files_edited
symbols_investigated
commands_run
tests_failing
tests_passing
bugs_found
fixes_attempted
user_constraints
pending_questions
known_risks
```

Also, compaction currently works mainly as a pressure response. For serious long sessions, Isaac should have incremental task memory, not just emergency summarization.

### Recommended direction

After each substantial turn, update a compact task checkpoint. Not necessarily by calling the model every time; some of it can be deterministic.

Example shape:

```json
{
  "active_goal": "Modernize ACP session restoration",
  "current_plan": [
    "Add brain restoration on session/load",
    "Persist structured file observations",
    "Add regression tests"
  ],
  "files": {
    "src/isaac/agent/acp/sessions.py": {
      "status": "edited",
      "last_known_sha256": "...",
      "important_symbols": ["load_session", "init_session"],
      "notes": ["load_session currently replays UI updates but does not restore PromptHandler state"]
    }
  },
  "commands": [
    {
      "cmd": "uv run pytest",
      "status": "failed",
      "important_output": "test_session_load_brain_restore failed before implementation"
    }
  ],
  "decisions": [
    "Brain restoration should happen automatically during session/load"
  ],
  "remaining_work": [
    "Wire restored state into PromptHandler",
    "Add cwd binding test for model-facing tools"
  ]
}
```

This would improve long-session reliability more than simply increasing the history limit.

---

## Critical issue 5: delegate subagents are useful but not fully integrated

Delegates are a good idea, especially for coding agents. But today they look too isolated from the parent session.

Potential issues:

1. Delegates appear to use the global/current model instead of the parent session’s model reliably.
2. Delegate tools likely suffer from the same cwd/session-context problem as main tools.
3. Inner delegate tool events are not fully surfaced to the parent’s event stream.
4. Parent history mostly receives a delegate summary, not concrete file observations/edits.
5. Recent-file tracking may not see files edited inside a delegate unless the delegate reports them correctly.

For a coding agent, this can create a dangerous illusion:

```text
Parent: delegated coding task completed.
Reality: parent does not have a reliable structured record of what changed, what was read, what failed, and what remains.
```

### Recommended fix

Delegate runs should receive the same session deps:

```text
session_id
cwd
additional_dirs
mode
model_id
permission callback
event observer
artifact recorder
```

And delegate outputs should be normalized into parent brain events:

```text
FileObservation
FileEdit
CommandResult
Decision
Risk
RemainingStep
```

Do not rely only on delegate prose.

---

## Other important correctness issues

### 1. `additional_directories` are ignored

In `sessions.py`, `additional_directories` are accepted but effectively ignored.

For ACP clients, this matters because they may provide multiple roots/context directories.

At minimum:

- validate them
- store them in session state
- pass them into tool safety/path resolution
- expose them to the model as available roots

### 2. Per-session model switching leaks through global model state

`_set_session_model()` appears to update both session state and a global current model.

That is risky for concurrent sessions.

A coding agent server should treat model choice as per-session unless the user explicitly changes a global default.

### 3. `close_session` likely leaks brain state

`close_session()` clears several ACP/session maps, but the prompt handler’s internal session state appears not to be cleared.

That can become a memory leak or stale-state bug.

### 4. `SessionStore.session_dir(session_id)` needs strict validation

If a client can pass arbitrary `session_id` into `load_session`, then `root / session_id` can become a path traversal risk unless IDs are validated.

Use UUID validation or a strict slug regex:

```python
SESSION_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,80}$")
```

Reject anything containing `/`, `\\`, `.`, or path separators.

### 5. Model-facing edit tools do not expose `expected_sha256`

The lower-level edit/apply-patch argument models seem to support optimistic concurrency, but the Pydantic wrapper signatures do not expose `expected_sha256`.

That means the model cannot naturally do the safe pattern:

```text
read file → get hash → edit with expected hash
```

For coding-agent correctness, expose this to the model-facing tools.

### 6. Tool output can still be too large

`read_file` and `code_search` need stricter model-visible output limits. Tool history summaries are truncated, but the model may still receive huge tool results during the live turn.

A coding agent should strongly prefer:

```text
read_file(path, start_line, end_line)
code_search(pattern, max_results)
```

and avoid dumping large files into context.

### 7. Plan progress is too loosely connected to real plan state

The current plan progress seems driven mostly by successful tool calls. That can mark progress even when a tool was just exploratory.

Better:

- plan steps should have stable IDs
- tool calls can be linked to step IDs
- the model or deterministic controller should mark step state explicitly

---

## Is the “brain” good enough for multi-step longer coding sessions?

Not yet.

It has the right ingredients, but the context strategy is still too shallow. For longer sessions, the core problem is not ACP compatibility. The core problem is maintaining a durable, queryable, task-aware working memory.

Right now Isaac has:

```text
recent chat history
some tool summaries
recent edited files
emergency compaction
manual checkpoint/restore
```

A stronger coding agent needs:

```text
durable session brain
automatic restore
file-aware observations
symbol/path/error retrieval
structured task checkpoint
safe workspace-bound tools
delegate artifact integration
```

So the current context engineering can be summarized as:

> Good foundations, but not yet robust enough for long, multi-step coding sessions where correctness depends on remembering earlier code discoveries and decisions.

---

## Recommended next implementation plan

### P0: Fix correctness foundations

These should be implemented before adding more agent-intelligence features.

#### P0.1 Bind all model-facing tools to session cwd/deps

Create a session-level dependency/context object and pass it to every model-facing tool wrapper.

Required fields:

```python
@dataclass(frozen=True)
class SessionToolDeps:
    session_id: str
    cwd: Path
    additional_directories: tuple[Path, ...]
    mode: PermissionMode
    model_id: str
    update_sink: ToolUpdateSink | None
```

Every model-triggered tool call should resolve paths through this context.

#### P0.2 Honor `additional_directories`

Implement full support for ACP-provided additional directories:

- validate paths
- store them in session state
- allow path resolution inside cwd or allowed additional roots
- reject everything else
- expose the available roots in model context

#### P0.3 Restore brain automatically on `session/load`

When loading a session:

1. Restore ACP UI updates.
2. Restore PromptHandler state.
3. Restore compact checkpoint.
4. Restore recent files.
5. Reconstruct useful model history if no direct brain snapshot exists.

#### P0.4 Validate session IDs before using them as paths

Use strict validation before passing IDs into filesystem paths.

Suggested rule:

```text
Only ASCII letters, digits, underscore, and dash.
Maximum length 80.
No dots.
No path separators.
```

#### P0.5 Clear PromptHandler state on `close_session`

When closing an ACP session, clean up all associated brain state as well as ACP state.

#### P0.6 Stop leaking per-session model selection into global model state

Model selection should be per-session. Avoid writing session-specific choices into global model state except when the user explicitly changes the default.

#### P0.7 Expose `expected_sha256` in model-facing edit/apply-patch tools

The model should be able to use optimistic concurrency safely:

```text
read file → get hash → edit with expected hash
```

#### P0.8 Add strict live tool-output caps

Do not only truncate history summaries. Also cap what the model sees during the active turn.

Especially constrain:

- `read_file`
- `code_search`
- `list_files`
- command output

---

### P1: Replace raw history trimming with a context selector

Instead of:

```python
history = trim_history(state.history, 30)
```

move toward:

```python
history = build_context_for_prompt(
    transcript=state.transcript,
    current_prompt=prompt,
    token_budget=model_context_budget,
)
```

The selector should include:

- current task checkpoint
- active plan
- last few turns
- relevant file observations
- recent edits
- recent test results
- user preferences/constraints
- compacted older decisions

Relevance should be based on:

- mentioned paths
- symbols
- test names
- error strings
- commands
- changed files
- current task summary
- explicit user references like “the previous bug” or “the design decision earlier”

---

### P2: Store structured coding memory

Persist events like:

```text
read file X lines A-B
searched symbol Y
edited file Z from hash H1 to H2
ran command C, failed with E
decision D
remaining step R
```

Suggested data model:

```python
@dataclass
class BrainEvent:
    id: str
    session_id: str
    turn_id: str
    kind: str
    timestamp: str
    payload: dict[str, object]
```

Useful event kinds:

```text
user_message
assistant_message
tool_call
tool_result
file_observation
file_edit
command_result
plan_update
decision
risk
open_question
checkpoint
```

This should survive restart and be selectable into future prompts.

---

### P3: Make delegates first-class brain participants

Delegates should not just return prose summaries. They should emit structured artifacts into the parent session:

```text
files_read
files_changed
commands_run
tests_result
risks
remaining_work
```

The parent session should consume those artifacts as ordinary brain events.

This prevents loss of information when delegated work is completed outside the main transcript.

---

## Suggested implementation sequence

### Step 1: Add `SessionToolDeps`

- Define the dependency object.
- Add it to prompt runner creation.
- Pass it into Pydantic AI agent runs.
- Update model-facing tool wrappers to use it.
- Add cwd correctness tests.

### Step 2: Implement workspace root validation

- Centralize path validation.
- Support cwd plus additional directories.
- Make tools reject paths outside allowed roots.
- Add tests for allowed and rejected paths.

### Step 3: Fix session load brain restoration

- Persist prompt state/checkpoint more explicitly.
- Load it during ACP `session/load`.
- Fall back to reconstructing brain history from persisted ACP updates.
- Add restart/load continuation tests.

### Step 4: Introduce structured brain events

- Start with minimal event types:
  - file observation
  - file edit
  - command result
  - decision
  - checkpoint
- Store them alongside existing history.
- Keep existing chat history for compatibility.

### Step 5: Replace fixed history trimming

- Build a context selector.
- First version can be simple heuristic retrieval.
- Prefer paths, symbols, test names, and error strings from current prompt.
- Include compact checkpoint and last few turns.

### Step 6: Upgrade delegates

- Pass session deps to delegate runs.
- Capture delegate artifacts as brain events.
- Surface important delegate tool progress to parent ACP stream.

### Step 7: Improve plan tracking

- Give plan steps stable IDs.
- Link tool calls to plan steps when possible.
- Add explicit plan-state updates.
- Avoid marking exploratory tool calls as plan progress automatically.

---

## Tests to add immediately

### 1. Session load brain continuity

```text
create session
ask agent to remember project-specific fact
persist/load session
ask follow-up referring to that fact
assert the model receives restored history/checkpoint
```

### 2. Model tool cwd correctness

```text
server process cwd != session cwd
model calls read_file("marker.txt")
assert marker comes from session cwd
```

### 3. Additional directory access

```text
session cwd = repo
additional dir = sibling docs
model can read allowed sibling docs
model cannot read unrelated paths
```

### 4. Expected hash is model-facing

```text
model-facing edit_file accepts expected_sha256
stale hash is rejected
```

### 5. Long context retention

```text
create 40 small observations across several files
ask about an old file-specific observation
assert context selector includes it
```

### 6. Delegate artifact propagation

```text
delegate edits file
parent recent files and transcript record the concrete changed file
```

### 7. Concurrent session model isolation

```text
session A selects model X
session B selects model Y
delegates/tools in A still use X
delegates/tools in B still use Y
```

### 8. Path traversal rejection

```text
load_session("../../evil")
assert rejected
```

---

## Definition of done for the next architecture milestone

The next milestone should not be “more tools” or “more prompt instructions.” It should be reliable long-session behavior.

A reasonable definition of done:

1. A loaded ACP session can continue with useful restored brain context.
2. All model-facing tools are bound to the correct session cwd and allowed roots.
3. `additional_directories` are honored safely.
4. The brain stores structured file/search/edit/command observations.
5. The context builder retrieves relevant old observations instead of blindly using the last 30 messages.
6. Delegates propagate structured artifacts back to the parent session.
7. Concurrent sessions do not leak model choice or workspace state into each other.
8. Regression tests cover restart/load, cwd correctness, and long-context retrieval.

---

## Final recommendation

Do not prioritize more intelligence features before fixing the foundation.

The project already has a respectable ACP/Pydantic AI skeleton. The next leap is not a better prompt. It is this:

> Turn the brain from a trimmed chat log into a durable, file-aware, task-aware coding memory.

Priority order:

1. Fix session cwd/tool deps.
2. Restore brain automatically on `session/load`.
3. Replace 30-message trimming with context selection.
4. Persist read/search/edit/test observations as structured events.
5. Make delegates report structured artifacts into parent memory.

After those changes, Isaac would be much closer to a serious coding agent rather than an ACP-compatible chat agent with tools.
