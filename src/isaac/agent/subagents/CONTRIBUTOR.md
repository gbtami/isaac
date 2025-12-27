Delegate Sub-Agent Tools
========================

This folder defines "delegate agent tools": small, specialized sub-agents that
the main Isaac agent can invoke as tools (for planning, review, coding, etc).
The goal is to make these easy to add and safe by default.

What this means
---------------
- A delegate tool is an ACP tool backed by a dedicated pydantic-ai Agent.
- Delegates run in isolation: they do NOT inherit the main chat history.
- Delegates receive only the task text (and an optional "carryover summary"
  from the same delegate tool when the caller passes a session_id and sets
  carryover=True).
- Delegates can have their own system prompt and tool allowlist.

Why this design
---------------
- Isolation avoids token bloat and role confusion.
- Specialized prompts/tools improve quality for focused tasks.
- A registry makes it easy to add new tools without editing many files.

How to add a new delegate tool
------------------------------
1) Create a new module in this folder, for example: `testing.py`.
2) Define:
   - A system prompt string (describe the role and expected output).
   - A structured output model (optional but recommended).
   - A `DelegateToolSpec` describing the tool name, prompt, and tool list.
   - A handler function that calls `run_delegate_tool`.
3) Register the tool at import time with `register_delegate_tool(...)`.

Minimal template
----------------
Use this pattern as a starting point:

```python
from isaac.agent.subagents.args import BaseDelegateArgs
from isaac.agent.subagents.delegate_tools import DelegateToolSpec, register_delegate_tool, run_delegate_tool
from isaac.agent.subagents.outputs import DelegateFileChange  # optional
from pydantic import BaseModel, Field

SYSTEM_PROMPT = """
You are Isaac's testing delegate.
Your job: run or describe tests needed for a change.
Return JSON that matches the output schema.
"""

class TestingDelegateOutput(BaseModel):
    summary: str = Field(..., description="Short testing summary.")
    tests: list[str] = Field(default_factory=list)

TOOL_SPEC = DelegateToolSpec(
    name="testing",
    description="Delegate test planning/execution.",
    instructions="Return JSON with a summary and tests list.",
    system_prompt=SYSTEM_PROMPT,
    tool_names=("list_files", "read_file", "run_command"),
    output_type=TestingDelegateOutput,
    log_context="delegate_testing",
)

async def testing(task: str, context: str | None = None, session_id: str | None = None, carryover: bool = False):
    return await run_delegate_tool(
        TOOL_SPEC,
        task=task,
        context=context,
        session_id=session_id,
        carryover=carryover,
    )

register_delegate_tool(TOOL_SPEC, handler=testing, arg_model=BaseDelegateArgs)
```

Important details
-----------------
- Naming: the tool name becomes the ACP tool name. Keep it short and clear.
- Args: new delegate tools must use `BaseDelegateArgs` (task/context/session_id/carryover).
  The registration wrapper in `tools/registration.py` assumes that signature.
- Tool list: keep it minimal; prefer read-only tools unless execution is required.
- Isolation: only the carryover summary is included when `carryover=True`.
  Full history is never passed.
- Timeouts and depth: set `timeout_s` and `max_depth` in `DelegateToolSpec` when needed.
- Output: prefer a structured output model to enable consistent summaries and tests.

Auto-discovery
--------------
Modules in this folder are imported at startup (see `subagents/__init__.py`).
As long as your module registers the tool on import, it will appear automatically
with no further code changes.
