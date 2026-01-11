"""Endpoint resolution for Google Code Assist."""

from __future__ import annotations

import os

# Code Assist uses sandbox (daily/autopush) plus prod endpoints. Requests prefer
# sandboxes for model traffic, while load/onboarding prefers prod for authoritative
# account/tier/project data (mirrors code-puppy behavior).
DEFAULT_CODE_ASSIST_ENDPOINTS = [
    "https://daily-cloudcode-pa.sandbox.googleapis.com",
    "https://autopush-cloudcode-pa.sandbox.googleapis.com",
    "https://cloudcode-pa.googleapis.com",
]
DEFAULT_CODE_ASSIST_LOAD_ENDPOINTS = [
    "https://cloudcode-pa.googleapis.com",
    "https://daily-cloudcode-pa.sandbox.googleapis.com",
    "https://autopush-cloudcode-pa.sandbox.googleapis.com",
]


def code_assist_endpoints() -> list[str]:
    env_list = os.getenv("ISAAC_CODE_ASSIST_ENDPOINTS")
    if env_list:
        return _normalize(env_list.split(","))

    env_single = os.getenv("ISAAC_CODE_ASSIST_ENDPOINT")
    if env_single:
        return _normalize([env_single])

    return _normalize(DEFAULT_CODE_ASSIST_ENDPOINTS)


def code_assist_load_endpoints() -> list[str]:
    env_list = os.getenv("ISAAC_CODE_ASSIST_LOAD_ENDPOINTS")
    if env_list:
        return _normalize(env_list.split(","))

    env_single = os.getenv("ISAAC_CODE_ASSIST_LOAD_ENDPOINT")
    if env_single:
        return _normalize([env_single])

    return _normalize(DEFAULT_CODE_ASSIST_LOAD_ENDPOINTS)


def primary_code_assist_endpoint() -> str:
    endpoints = code_assist_endpoints()
    return endpoints[0] if endpoints else "https://daily-cloudcode-pa.sandbox.googleapis.com"


def _normalize(values: list[str]) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for value in values:
        endpoint = value.strip().rstrip("/")
        if not endpoint or endpoint in seen:
            continue
        seen.add(endpoint)
        normalized.append(endpoint)
    return normalized
