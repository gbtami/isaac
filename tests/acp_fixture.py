"""Real stdio server with a deterministic model; never calls a provider API."""

import asyncio
import json
import sys
from collections.abc import AsyncIterator
from uuid import uuid4

from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from pydantic_ai_harness.experimental.acp import run_acp_stdio

from isaac.config import Settings
from isaac.server import build_agent, configure_session


async def stream(messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
    """Call the tool named by a JSON user prompt, then return its output."""
    if isinstance(messages[-1], ModelRequest):
        for part in reversed(messages[-1].parts):
            if isinstance(part, ToolReturnPart):
                yield str(part.content)
                return
    for message in reversed(messages):
        if not isinstance(message, ModelRequest):
            continue
        for part in reversed(message.parts):
            if not isinstance(part, UserPromptPart):
                continue
            contents = [part.content] if isinstance(part.content, str) else part.content
            for content in contents:
                if not isinstance(content, str) or not content.startswith('{"tool":'):
                    continue
                command = json.loads(content)
                tool = command["tool"]
                arguments = command["args"]
                if tool == "history_length":
                    yield str(len(messages))
                    return
                if tool not in {definition.name for definition in _info.function_tools}:
                    keywords = ", ".join(f"{key}={value!r}" for key, value in arguments.items())
                    arguments = {"code": f"print(await {tool}({keywords}))"}
                    tool = "run_code"
                yield {0: DeltaToolCall(name=tool, json_args=json.dumps(arguments), tool_call_id=uuid4().hex)}
                return
    yield "ready"


def summarize(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
    if "--fail-summary" in sys.argv:
        raise ModelHTTPError(status_code=503, model_name="fixture", body="Summary model unavailable.")
    return ModelResponse(parts=[TextPart("Fixture summary.")])


async def main() -> None:
    settings = Settings(
        models=("test:fixture",), permissions="read-only", skill_dirs=(), code_mode="--code-mode" in sys.argv
    )
    agent = build_agent(
        settings,
        model=FunctionModel(
            function=summarize,
            stream_function=stream,
        ),
    )
    async with agent:
        await run_acp_stdio(agent, name="isaac-fixture", session_config=configure_session)


if __name__ == "__main__":
    asyncio.run(main())
