"""Auth method helpers for ACP client initialization/authentication."""

from __future__ import annotations

import sys
from typing import Any

from acp.schema import AuthMethod


def extract_auth_methods(init_resp: Any) -> list[AuthMethod | Any]:
    auth_methods = getattr(init_resp, "auth_methods", None)
    if auth_methods is None:
        auth_methods = getattr(init_resp, "authMethods", None)
    if auth_methods is None and isinstance(init_resp, dict):
        auth_methods = init_resp.get("auth_methods") or init_resp.get("authMethods")
    if not isinstance(auth_methods, list):
        return []
    methods: list[AuthMethod | Any] = []
    for method in auth_methods:
        method_id = _auth_method_id(method)
        if method_id:
            methods.append(method)
    return methods


def extract_error_auth_methods(error_data: Any) -> list[AuthMethod | Any]:
    if not isinstance(error_data, dict):
        return []
    return extract_auth_methods(
        {
            "auth_methods": error_data.get("auth_methods"),
            "authMethods": error_data.get("authMethods"),
        }
    )


def select_auth_method(auth_methods: list[AuthMethod | Any]) -> str:
    if not auth_methods:
        raise ValueError("No authentication methods available.")
    if len(auth_methods) == 1 or not sys.stdin.isatty():
        return _auth_method_id(auth_methods[0])

    print("Agent requires authentication. Available methods:")
    for idx, method in enumerate(auth_methods, start=1):
        method_id = _auth_method_id(method)
        method_name = _auth_method_name(method) or method_id
        method_description = _auth_method_description(method)
        label = f"{idx}) {method_name} ({method_id})"
        if method_description:
            label = f"{label} - {method_description}"
        print(label)

    selection = input("Authentication method (number): ").strip()
    if selection.isdigit():
        index = int(selection)
        if 1 <= index <= len(auth_methods):
            return _auth_method_id(auth_methods[index - 1])
    return _auth_method_id(auth_methods[0])


def find_auth_method(auth_methods: list[AuthMethod | Any], method_id: str) -> AuthMethod | Any | None:
    target = method_id.strip().lower()
    for method in auth_methods:
        if _auth_method_id(method).strip().lower() == target:
            return method
    return None


def auth_method_type(auth_method: AuthMethod | Any) -> str:
    # ACP treats "agent" as the default type for backward compatibility.
    payload_type = _payload_value(auth_method, "type")
    meta_type = _auth_method_meta(auth_method).get("type")
    value = str(payload_type or meta_type or "agent").strip().lower()
    return value or "agent"


def auth_method_env_var_name(auth_method: AuthMethod | Any) -> str | None:
    value = _payload_value(auth_method, "varName", "var_name")
    if not value:
        meta = _auth_method_meta(auth_method)
        value = meta.get("varName") or meta.get("var_name")
    env_name = str(value or "").strip()
    return env_name or None


def auth_method_link(auth_method: AuthMethod | Any) -> str | None:
    value = _payload_value(auth_method, "link")
    if not value:
        value = _auth_method_meta(auth_method).get("link")
    link = str(value or "").strip()
    return link or None


def _auth_method_id(auth_method: AuthMethod | Any) -> str:
    if isinstance(auth_method, dict):
        return str(auth_method.get("id", "")).strip()
    return str(getattr(auth_method, "id", "")).strip()


def _auth_method_name(auth_method: AuthMethod | Any) -> str:
    if isinstance(auth_method, dict):
        return str(auth_method.get("name", "")).strip()
    return str(getattr(auth_method, "name", "")).strip()


def _auth_method_description(auth_method: AuthMethod | Any) -> str:
    if isinstance(auth_method, dict):
        return str(auth_method.get("description", "") or "").strip()
    return str(getattr(auth_method, "description", "") or "").strip()


def _payload_value(auth_method: AuthMethod | Any, *keys: str) -> Any:
    if isinstance(auth_method, dict):
        for key in keys:
            if key in auth_method:
                return auth_method.get(key)
        return None
    for key in keys:
        value = getattr(auth_method, key, None)
        if value is not None:
            return value
    return None


def _auth_method_meta(auth_method: AuthMethod | Any) -> dict[str, Any]:
    if isinstance(auth_method, dict):
        raw = auth_method.get("_meta")
        return raw if isinstance(raw, dict) else {}
    raw = getattr(auth_method, "_meta", None)
    return raw if isinstance(raw, dict) else {}
