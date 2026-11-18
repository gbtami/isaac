"""Real stdio server with a deterministic model; never calls a provider API."""

import asyncio
import json
import sys
from collections.abc import AsyncIterator
from functools import partial
from uuid import uuid4

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
from pydantic_ai_harness.experimental.acp import run_acp_stdio

from isaac.config import Settings, session_database
from isaac.server import build_agent, configure_session
from isaac.session_store import SqliteSessionStore


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
                    arguments = {"code": f"print(await {tool}({keywords}))"}
                    tool = "run_code"
                yield {0: DeltaToolCall(name=tool, json_args=json.dumps(arguments), tool_call_id=uuid4().hex)}
                return
    yield "ready"


def summarize(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
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
        code_mode="--code-mode" in sys.argv,
    )
    model_class = NativeCompactionModel if "--native-compact" in sys.argv else FunctionModel
    model = model_class(function=summarize, stream_function=stream)
    agent = build_agent(settings, model=model)
    store = SqliteSessionStore(session_database())
    async with agent:
        await run_acp_stdio(
            agent,
            name="isaac-fixture",
            session_config=partial(configure_session, store=store),
            session_store=store,
            models=settings.models,
            model_resolver=lambda _name: model,
        )


if __name__ == "__main__":
    asyncio.run(main())
