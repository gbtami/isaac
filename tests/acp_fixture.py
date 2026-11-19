"""Real stdio server with a deterministic model; never calls a provider API."""

import asyncio
import json
import sys
from collections.abc import AsyncIterator
from dataclasses import replace
from functools import partial
from pathlib import Path
from uuid import uuid4

import acp
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import (
    CompactionPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from pydantic_ai.usage import RequestUsage
from pydantic_ai_harness.sub_agents import is_task_report

from isaac.config import Settings
from isaac.server import build_acp_adapter, build_agent


async def stream(
    messages: list[ModelMessage], _info: AgentInfo, *, model_id: str
) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
    """Call the tool named by a JSON user prompt, then return its output."""
    if "fixture_named_agent_instructions" in (_info.instructions or ""):
        yield f"Named delegate used {model_id}"
        return
    if isinstance(messages[-1], ModelRequest):
        for part in reversed(messages[-1].parts):
            if isinstance(part, ToolReturnPart):
                if part.tool_name in {"read_file", "run_code"} and "checkpoint evidence" in str(part.content):
                    tool = "run_code" if part.tool_name == "run_code" else "checkpoint_probe"
                    args = {"code": "await checkpoint_probe()"} if tool == "run_code" else {}
                    yield {0: DeltaToolCall(name=tool, json_args=json.dumps(args), tool_call_id="checkpoint-probe")}
                    return
                yield str(part.content)
                return
        if any(is_task_report(part) for part in messages[-1].parts):
            yield "Task report received."
            return
    for message in reversed(messages):
        if not isinstance(message, ModelRequest):
            continue
        for part in reversed(message.parts):
            if not isinstance(part, UserPromptPart):
                continue
            contents = [part.content] if isinstance(part.content, str) else part.content
            for content in contents:
                if content == "fixture block until cancelled":
                    await asyncio.Future[None]()
                    return
                if content == "fixture continue from checkpoint":
                    returns = [
                        part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)
                    ]
                    if any("checkpoint evidence" in str(part.content) for part in returns) and any(
                        part.tool_call_id == "checkpoint-probe"
                        and (part.metadata or {}).get("pydantic_ai_synthesized_tool_return")
                        for part in returns
                    ):
                        yield "Restored evidence and closed the unfinished call."
                    else:
                        yield "Checkpoint context missing."
                    return
                if not isinstance(content, str) or not content.startswith('{"tool":'):
                    continue
                command = json.loads(content)
                tool = command["tool"]
                arguments = command["args"]
                if tool == "history_length":
                    yield str(len(messages))
                    return
                if tool == "model_name":
                    yield model_id
                    return
                if tool == "check_compaction":
                    restored = any(
                        isinstance(part, CompactionPart)
                        and (part.provider_details or {}).get("encrypted_content") == "fixture-opaque-state"
                        for previous in messages
                        for part in previous.parts
                    )
                    yield "compaction restored" if restored else "compaction missing"
                    return
                if tool not in {definition.name for definition in _info.function_tools}:
                    keywords = ", ".join(f"{key}={value!r}" for key, value in arguments.items())
                    call = f"{tool}({keywords})"
                    if tool == "write_plan":
                        call = f"await {call}"
                    arguments = {"code": f"print({call})"}
                    tool = "run_code"
                yield {0: DeltaToolCall(name=tool, json_args=json.dumps(arguments), tool_call_id=uuid4().hex)}
                return
    yield "ready"


def summarize(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
    if "fixture_named_agent_instructions" in (_info.instructions or ""):
        return ModelResponse(parts=[TextPart("Named delegate used its disk instructions.")])
    if any(
        "concise title" in (message.instructions or "") for message in _messages if isinstance(message, ModelRequest)
    ):
        if "--fail-title" in sys.argv:
            raise ModelHTTPError(status_code=503, model_name="fixture", body="Title model unavailable.")
        return ModelResponse(parts=[TextPart("Improve session persistence")])
    if "--fail-summary" in sys.argv:
        raise ModelHTTPError(status_code=503, model_name="fixture", body="Summary model unavailable.")
    return ModelResponse(parts=[TextPart("Fixture summary.")])


class NativeCompactionModel(FunctionModel):
    async def compact_messages(
        self, request_context: ModelRequestContext, *, instructions: str | None = None
    ) -> ModelResponse:
        assert isinstance(request_context.messages[-1], ModelResponse), "Exclude the /compact command from history"
        if "--fail-native-compact" in sys.argv:
            raise ModelHTTPError(status_code=404, model_name=self.model_name, body="Compaction unavailable.")
        return ModelResponse(
            parts=[
                CompactionPart(
                    provider_name=self.system, provider_details={"encrypted_content": "fixture-opaque-state"}
                )
            ],
            provider_name=self.system,
            usage=RequestUsage(input_tokens=10, output_tokens=3),
        )


async def main() -> None:
    settings = Settings(
        models=("fixture:primary", "fixture:secondary"),
        permissions="read-only",
        skill_dirs=(),
        context_window=50_000 if "--known-context-window" in sys.argv else None,
        agent_dirs=(".agents/agents",) if "--named-delegates" in sys.argv else (),
    )
    if "--no-code-mode" in sys.argv:
        settings = replace(settings, code_mode=False)
    model_class = NativeCompactionModel if "--native-compact" in sys.argv else FunctionModel
    models = {
        name: model_class(function=summarize, stream_function=partial(stream, model_id=name), model_name=name)
        for name in settings.models
    }
    agent = build_agent(settings)
    if "--checkpoint-probe" in sys.argv:
        effect_path = Path(sys.argv[sys.argv.index("--checkpoint-probe") + 1])

        @agent.tool_plain
        async def checkpoint_probe() -> None:
            """Record a side effect, then wait for the test to terminate the server."""
            with effect_path.open("a") as file:
                file.write("started\n")
            await asyncio.Future[None]()

    adapter = build_acp_adapter(settings, agent, resolve_model=models.__getitem__)
    async with agent:
        await acp.run_agent(adapter, use_unstable_protocol=True)


if __name__ == "__main__":
    asyncio.run(main())
