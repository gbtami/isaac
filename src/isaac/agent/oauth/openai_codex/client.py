"""HTTP client adapter for ChatGPT Codex API requirements."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


def _is_reasoning_model(model_name: str) -> bool:
    if not model_name:
        return False
    model_lower = model_name.lower()
    return model_lower.startswith(("gpt-5", "o1", "o3", "o4"))


class OpenAICodexAsyncClient(httpx.AsyncClient):
    """Async HTTP client that injects Codex-required fields."""

    async def send(self, request: httpx.Request, *args: Any, **kwargs: Any) -> httpx.Response:
        force_stream_conversion = False

        try:
            if request.method == "POST":
                body_bytes = _extract_body_bytes(request)
                if body_bytes:
                    updated, force_stream_conversion = _inject_codex_fields(body_bytes)
                    if updated is not None:
                        rebuilt = self.build_request(
                            method=request.method,
                            url=request.url,
                            headers=request.headers,
                            content=updated,
                        )
                        if hasattr(rebuilt, "_content"):
                            setattr(request, "_content", rebuilt._content)
                        if hasattr(rebuilt, "stream"):
                            request.stream = rebuilt.stream
                        if hasattr(rebuilt, "extensions"):
                            request.extensions = rebuilt.extensions
                        request.headers["Content-Length"] = str(len(updated))
        except Exception:
            pass

        response = await super().send(request, *args, **kwargs)

        if force_stream_conversion and response.status_code == 200:
            try:
                response = await _convert_stream_to_response(response)
            except Exception as exc:
                logger.warning("Failed to convert Codex stream response: %s", exc)

        return response


def _extract_body_bytes(request: httpx.Request) -> bytes | None:
    try:
        content = request.content
        if content:
            return content
    except Exception:
        pass

    try:
        content = getattr(request, "_content", None)
        if content:
            return content
    except Exception:
        pass

    return None


def _inject_codex_fields(body: bytes) -> tuple[bytes | None, bool]:
    try:
        data = json.loads(body.decode("utf-8"))
    except Exception:
        return None, False

    if not isinstance(data, dict):
        return None, False

    modified = False
    forced_stream = False

    if data.get("store") is not False:
        data["store"] = False
        modified = True

    if data.get("stream") is not True:
        data["stream"] = True
        forced_stream = True
        modified = True

    model = data.get("model", "")
    if "reasoning" not in data and _is_reasoning_model(str(model)):
        data["reasoning"] = {"effort": "medium", "summary": "auto"}
        modified = True

    for key in ("max_output_tokens", "max_tokens", "verbosity"):
        if key in data:
            del data[key]
            modified = True

    if not modified:
        return None, False

    return json.dumps(data).encode("utf-8"), forced_stream


async def _convert_stream_to_response(response: httpx.Response) -> httpx.Response:
    final_response_data = None
    collected_text: list[str] = []
    collected_tool_calls: list[dict[str, Any]] = []

    async for line in response.aiter_lines():
        if not line or not line.startswith("data:"):
            continue
        data_str = line[5:].strip()
        if data_str == "[DONE]":
            break

        try:
            event = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        event_type = event.get("type", "")
        if event_type == "response.output_text.delta":
            delta = event.get("delta", "")
            if delta:
                collected_text.append(delta)
        elif event_type == "response.completed":
            final_response_data = event.get("response", {})
        elif event_type == "response.function_call_arguments.done":
            collected_tool_calls.append(
                {
                    "name": event.get("name", ""),
                    "arguments": event.get("arguments", ""),
                    "call_id": event.get("call_id", ""),
                }
            )

    if final_response_data:
        response_body = final_response_data
    else:
        response_body = {
            "id": "reconstructed",
            "object": "response",
            "output": [],
        }

        if collected_text:
            response_body["output"].append(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "".join(collected_text)}],
                }
            )

        for tool_call in collected_tool_calls:
            response_body["output"].append(
                {
                    "type": "function_call",
                    "name": tool_call["name"],
                    "arguments": tool_call["arguments"],
                    "call_id": tool_call["call_id"],
                }
            )

    body_bytes = json.dumps(response_body).encode("utf-8")
    return httpx.Response(
        status_code=response.status_code,
        headers=response.headers,
        content=body_bytes,
        request=response.request,
    )
