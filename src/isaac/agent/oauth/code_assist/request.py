"""Request shaping helpers for Google Code Assist."""

from __future__ import annotations

import base64
import uuid
from typing import Any

DEFAULT_MAX_OUTPUT_TOKENS = 64000
DEFAULT_TOP_K = 64
DEFAULT_TOP_P = 0.95


def apply_code_assist_envelope(
    payload: dict[str, Any],
    model_name: str,
    project_id: str,
) -> dict[str, Any]:
    request = payload.get("request")
    if isinstance(request, dict):
        _strip_system_role(request)
        _normalize_tools(request)
        _apply_generation_defaults(request)
        if "sessionId" in request and "session_id" not in request:
            request["session_id"] = request.pop("sessionId")
        request.setdefault("session_id", _build_session_id(model_name, project_id))

    payload.setdefault("project", project_id)
    return _json_safe(payload)


def _build_session_id(model_name: str, project_id: str) -> str:
    seed = uuid.uuid4().hex[:16]
    return f"-{uuid.uuid4()}:{model_name}:{project_id}:seed-{seed}"


def _strip_system_role(request: dict[str, Any]) -> None:
    system_instruction = request.get("systemInstruction")
    if isinstance(system_instruction, dict) and "role" in system_instruction:
        system_instruction.pop("role", None)


def _apply_generation_defaults(request: dict[str, Any]) -> None:
    generation = request.get("generationConfig")
    if not isinstance(generation, dict):
        generation = {}
    generation.setdefault("topK", DEFAULT_TOP_K)
    generation.setdefault("topP", DEFAULT_TOP_P)
    generation.setdefault("maxOutputTokens", DEFAULT_MAX_OUTPUT_TOKENS)
    if generation:
        request["generationConfig"] = generation


def _normalize_tools(request: dict[str, Any]) -> None:
    tools = request.get("tools")
    if not isinstance(tools, list):
        return
    function_decls: list[dict[str, Any]] = []
    other_tools: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        declarations = tool.get("functionDeclarations")
        if not isinstance(declarations, list):
            declarations = tool.get("function_declarations")
        if isinstance(declarations, list):
            for declaration in declarations:
                if not isinstance(declaration, dict):
                    continue
                if "parameters" not in declaration:
                    if "parametersJsonSchema" in declaration:
                        declaration["parameters"] = declaration.pop("parametersJsonSchema")
                    elif "parameters_json_schema" in declaration:
                        declaration["parameters"] = declaration.pop("parameters_json_schema")
                function_decls.append(declaration)
            continue
        non_function = {
            key: value for key, value in tool.items() if key not in {"functionDeclarations", "function_declarations"}
        }
        if non_function:
            other_tools.append(non_function)

    if function_decls:
        request["tools"] = [{"functionDeclarations": function_decls}]
        return
    if other_tools:
        request["tools"] = other_tools[:1]


def _json_safe(value: Any) -> Any:
    """Recursively coerce provider payload values to JSON-serializable types.

    pydantic-ai may pass bytes (e.g. thought signatures) in Google message parts.
    The Code Assist HTTP path uses httpx ``json=...`` and must not contain raw
    bytes, so encode binary values as base64 strings.
    """

    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (bytes, bytearray, memoryview)):
        return base64.b64encode(bytes(value)).decode("ascii")
    return value
