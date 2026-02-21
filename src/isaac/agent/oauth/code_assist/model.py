"""pydantic-ai model wrapper for Google Code Assist."""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Iterable, cast

from google.genai import types
from pydantic_ai import usage
from pydantic_ai.models import (
    Model,
    ModelRequestParameters,
    ModelResponse,
    StreamedResponse,
    check_allow_model_requests,
)
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.providers import Provider
from pydantic_ai.profiles.google import google_model_profile

from isaac.agent.oauth.code_assist.auth import ensure_project, get_access_token
from isaac.agent.oauth.code_assist.client import CODE_ASSIST_ENDPOINT, CodeAssistClient
from isaac.agent.oauth.code_assist.request import apply_code_assist_envelope


class CodeAssistProvider(Provider[object]):
    def __init__(self, base_url: str) -> None:
        self._base_url = base_url
        self._client = object()

    @property
    def name(self) -> str:
        return "google-code-assist"

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def client(self) -> object:
        return self._client

    def model_profile(self, model_name: str):  # type: ignore[override]
        return google_model_profile(model_name)


class CodeAssistModel(Model):
    """Model implementation that targets the Code Assist backend."""

    def __init__(
        self,
        model_name: str,
        *,
        profile: Any | None = None,
        settings: Any | None = None,
        client: CodeAssistClient | None = None,
    ) -> None:
        self._model_name = model_name
        self._client = client or CodeAssistClient()
        self._provider = CodeAssistProvider(base_url=CODE_ASSIST_ENDPOINT)
        self._google_helper = GoogleModel(
            model_name,
            provider=self._provider,
            profile=profile,
            settings=settings,
        )
        super().__init__(settings=settings, profile=profile or google_model_profile)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def system(self) -> str:
        return "google-code-assist"

    @property
    def base_url(self) -> str:
        return self._provider.base_url

    @classmethod
    def supported_builtin_tools(cls) -> frozenset[type]:  # type: ignore[override]
        return GoogleModel.supported_builtin_tools()

    async def request(
        self,
        messages: list[Any],
        model_settings: Any | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        check_allow_model_requests()
        model_settings, model_request_parameters = self._google_helper.prepare_request(
            model_settings, model_request_parameters
        )
        model_settings = cast(GoogleModelSettings, model_settings or {})
        contents, config = await self._google_helper._build_content_and_config(  # noqa: SLF001
            messages, model_settings, model_request_parameters
        )
        request_payload = await _build_request_payload(self._model_name, contents, config, model_request_parameters)
        tokens = await get_access_token()
        tokens = await ensure_project(tokens)
        request_payload["project"] = tokens.project_id
        request_payload = apply_code_assist_envelope(request_payload, self._model_name, tokens.project_id)
        response_data = await self._client.post_method(
            "generateContent",
            request_payload,
            tokens.access_token,
            model_name=self._model_name,
        )
        response_payload = response_data.get("response") or {}
        if response_data.get("traceId") and "responseId" not in response_payload:
            response_payload["responseId"] = response_data["traceId"]
        response = types.GenerateContentResponse.model_validate(response_payload)
        return self._google_helper._process_response(response)  # noqa: SLF001

    async def count_tokens(
        self,
        messages: list[Any],
        model_settings: Any | None,
        model_request_parameters: ModelRequestParameters,
    ) -> usage.RequestUsage:
        check_allow_model_requests()
        model_settings, model_request_parameters = self._google_helper.prepare_request(
            model_settings, model_request_parameters
        )
        model_settings = cast(GoogleModelSettings, model_settings or {})
        contents, _config = await self._google_helper._build_content_and_config(  # noqa: SLF001
            messages, model_settings, model_request_parameters
        )
        payload = {
            "request": {
                "model": f"models/{self._model_name}",
                "contents": _dump_contents(contents),
            }
        }
        tokens = await get_access_token()
        tokens = await ensure_project(tokens)
        response = await self._client.post_method(
            "countTokens",
            payload,
            tokens.access_token,
            model_name=self._model_name,
        )
        total_tokens = response.get("totalTokens")
        if total_tokens is None:
            raise RuntimeError("Code Assist countTokens response missing totalTokens.")
        return usage.RequestUsage(input_tokens=int(total_tokens))

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[Any],
        model_settings: Any | None,
        model_request_parameters: ModelRequestParameters,
        run_context: Any | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        check_allow_model_requests()
        model_settings, model_request_parameters = self._google_helper.prepare_request(
            model_settings, model_request_parameters
        )
        model_settings = cast(GoogleModelSettings, model_settings or {})
        contents, config = await self._google_helper._build_content_and_config(  # noqa: SLF001
            messages, model_settings, model_request_parameters
        )
        request_payload = await _build_request_payload(self._model_name, contents, config, model_request_parameters)
        tokens = await get_access_token()
        tokens = await ensure_project(tokens)
        request_payload["project"] = tokens.project_id
        request_payload = apply_code_assist_envelope(request_payload, self._model_name, tokens.project_id)

        async def _iter_responses() -> AsyncIterator[types.GenerateContentResponse]:
            async for item in self._client.stream_method(
                "streamGenerateContent",
                request_payload,
                tokens.access_token,
                model_name=self._model_name,
            ):
                response_payload = item.get("response") or {}
                if item.get("traceId") and "responseId" not in response_payload:
                    response_payload["responseId"] = item["traceId"]
                yield types.GenerateContentResponse.model_validate(response_payload)

        streamed = await self._google_helper._process_streamed_response(  # noqa: SLF001
            _iter_responses(), model_request_parameters
        )
        yield streamed


async def _build_request_payload(
    model_name: str,
    contents: list[dict[str, Any]],
    config: dict[str, Any],
    model_request_parameters: ModelRequestParameters,
) -> dict[str, Any]:
    contents_payload = _dump_contents(contents)
    config_payload = _dump_config(config)
    generation_config: dict[str, Any] = {}
    for key in list(config_payload.keys()):
        if key in _GENERATION_KEYS:
            generation_config[key] = config_payload.pop(key)
    for key in _UNSUPPORTED_GENERATION_KEYS:
        generation_config.pop(key, None)
    if model_request_parameters.allow_text_output is False:
        generation_config.setdefault("responseMimeType", "application/json")

    request_payload: dict[str, Any] = {"contents": contents_payload}
    if "systemInstruction" in config_payload:
        request_payload["systemInstruction"] = config_payload["systemInstruction"]
    if "tools" in config_payload:
        request_payload["tools"] = config_payload["tools"]
    if generation_config:
        request_payload["generationConfig"] = generation_config
    return {
        "model": model_name,
        "user_prompt_id": str(uuid.uuid4()),
        "request": {
            **request_payload,
        },
    }


def _dump_contents(contents: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    dumped: list[dict[str, Any]] = []
    for content in contents:
        dumped.append(types.Content.model_validate(content).model_dump(by_alias=True, exclude_none=True))
    return dumped


def _dump_config(config: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(config)
    cleaned.pop("http_options", None)
    cleaned.pop("should_return_http_response", None)
    dumped = types.GenerateContentConfig.model_validate(cleaned).model_dump(by_alias=True, exclude_none=True)
    return {
        key: value
        for key, value in dumped.items()
        if key
        not in {
            "httpOptions",
            "shouldReturnHttpResponse",
        }
    }


_GENERATION_KEYS = {
    "temperature",
    "topP",
    "topK",
    "candidateCount",
    "maxOutputTokens",
    "stopSequences",
    "responseLogprobs",
    "logprobs",
    "presencePenalty",
    "frequencyPenalty",
    "seed",
    "responseMimeType",
    "responseJsonSchema",
    "responseSchema",
    "routingConfig",
    "modelSelectionConfig",
    "responseModalities",
    "mediaResolution",
    "speechConfig",
    "audioTimestamp",
    "thinkingConfig",
}

_UNSUPPORTED_GENERATION_KEYS = {
    "responseModalities",
}
