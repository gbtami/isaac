"""HTTP client for Google Code Assist APIs."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
from importlib.metadata import PackageNotFoundError, version
from typing import Any, AsyncIterator

import httpx

from isaac.agent.oauth.code_assist.endpoints import code_assist_endpoints, primary_code_assist_endpoint

CODE_ASSIST_ENDPOINTS = code_assist_endpoints()
CODE_ASSIST_ENDPOINT = primary_code_assist_endpoint()
CODE_ASSIST_API_VERSION = os.getenv("ISAAC_CODE_ASSIST_API_VERSION", "v1internal")
CODE_ASSIST_RATE_LIMIT_RETRIES = int(os.getenv("ISAAC_CODE_ASSIST_RATE_LIMIT_RETRIES", "5"))
CODE_ASSIST_RATE_LIMIT_MAX_WAIT_S = float(os.getenv("ISAAC_CODE_ASSIST_RATE_LIMIT_MAX_WAIT_S", "60"))
CODE_ASSIST_HEADER_STYLES_ENV = "ISAAC_CODE_ASSIST_HEADER_STYLES"
DEFAULT_CODE_ASSIST_HEADER_STYLE = "gemini-cli"

_FALLBACK_STATUSES = {403, 404, 408, 429, 500, 502, 503, 504}
_RATE_LIMIT_MESSAGE_MAX = 240
logger = logging.getLogger(__name__)


class CodeAssistClient:
    def __init__(
        self,
        http_client: httpx.AsyncClient | None = None,
        base_urls: list[str] | None = None,
    ) -> None:
        self._client = http_client or httpx.AsyncClient(timeout=30)
        endpoints = base_urls or CODE_ASSIST_ENDPOINTS
        if not endpoints:
            raise RuntimeError("No Code Assist endpoints configured.")
        self._base_urls = [f"{endpoint}/{CODE_ASSIST_API_VERSION}" for endpoint in endpoints]
        self._base_url = self._base_urls[0]

    @property
    def base_url(self) -> str:
        return self._base_url

    async def post_method(
        self,
        method: str,
        payload: dict[str, Any],
        access_token: str,
        model_name: str | None = None,
    ) -> dict[str, Any]:
        header_style = _header_style()
        last_exc: Exception | None = None
        rate_limit_error: Exception | None = None
        for base_url in self._base_urls:
            url = f"{base_url}:{method}"
            try:
                advance_endpoint = False
                headers = self._headers(access_token, model_name=model_name)
                last_response: httpx.Response | None = None
                for attempt in range(CODE_ASSIST_RATE_LIMIT_RETRIES):
                    response = await self._client.post(url, json=payload, headers=headers)
                    last_response = response
                    if response.status_code == 429:
                        delay, summary, error_info = await _parse_rate_limit_error(response)
                        if summary:
                            logger.warning(
                                "Code Assist rate limited (attempt %s/%s, style=%s): %s",
                                attempt + 1,
                                CODE_ASSIST_RATE_LIMIT_RETRIES,
                                header_style,
                                summary,
                            )
                        if _should_abort_rate_limit(error_info, delay):
                            rate_limit_error = RuntimeError(
                                "Code Assist quota exhausted (RESOURCE_EXHAUSTED). "
                                "Try again later or use a different account."
                            )
                            advance_endpoint = True
                            break
                        retry_delay = delay if delay is not None else 2.0
                        if retry_delay < CODE_ASSIST_RATE_LIMIT_MAX_WAIT_S:
                            await asyncio.sleep(retry_delay + 0.1)
                            continue
                    if response.status_code >= 400:
                        await _log_http_error(response, method, header_style)
                    response.raise_for_status()
                    return response.json()
                if advance_endpoint:
                    logger.warning(
                        "Code Assist rate limit exhaustion on %s (style=%s); trying next endpoint.",
                        base_url,
                        header_style,
                    )
                    continue
                if last_response is not None:
                    last_response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                if not _should_fallback(exc):
                    raise
                last_exc = exc
            except httpx.HTTPError as exc:
                last_exc = exc
        if rate_limit_error:
            raise rate_limit_error
        if last_exc:
            raise last_exc
        raise RuntimeError("Code Assist request failed without an error response.")

    async def stream_method(
        self,
        method: str,
        payload: dict[str, Any],
        access_token: str,
        model_name: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        header_style = _header_style()
        last_exc: Exception | None = None
        rate_limit_error: Exception | None = None
        for base_url in self._base_urls:
            url = f"{base_url}:{method}"
            try:
                advance_endpoint = False
                headers = self._headers(access_token, model_name=model_name)
                last_response: httpx.Response | None = None
                for attempt in range(CODE_ASSIST_RATE_LIMIT_RETRIES):
                    async with self._client.stream(
                        "POST",
                        url,
                        params={"alt": "sse"},
                        headers={**headers, "Accept": "text/event-stream"},
                        json=payload,
                    ) as response:
                        last_response = response
                        if response.status_code == 429:
                            delay, summary, error_info = await _parse_rate_limit_error(response)
                            if summary:
                                logger.warning(
                                    "Code Assist rate limited (attempt %s/%s, style=%s): %s",
                                    attempt + 1,
                                    CODE_ASSIST_RATE_LIMIT_RETRIES,
                                    header_style,
                                    summary,
                                )
                            if _should_abort_rate_limit(error_info, delay):
                                rate_limit_error = RuntimeError(
                                    "Code Assist quota exhausted (RESOURCE_EXHAUSTED). "
                                    "Try again later or use a different account."
                                )
                                advance_endpoint = True
                                break
                            retry_delay = delay if delay is not None else 2.0
                            if retry_delay < CODE_ASSIST_RATE_LIMIT_MAX_WAIT_S:
                                await asyncio.sleep(retry_delay + 0.1)
                                continue
                        if response.status_code >= 400:
                            await _log_http_error(response, method, header_style)
                        response.raise_for_status()
                        async for item in _iter_sse(response):
                            yield item
                        return
                if advance_endpoint:
                    logger.warning(
                        "Code Assist rate limit exhaustion on %s (style=%s); trying next endpoint.",
                        base_url,
                        header_style,
                    )
                    continue
                if last_response is not None:
                    raise httpx.HTTPStatusError(
                        "Rate limited",
                        request=last_response.request,
                        response=last_response,
                    )
            except httpx.HTTPStatusError as exc:
                if not _should_fallback(exc):
                    raise
                last_exc = exc
            except httpx.HTTPError as exc:
                last_exc = exc
        if rate_limit_error:
            raise rate_limit_error
        if last_exc:
            raise last_exc
        raise RuntimeError("Code Assist stream failed without an error response.")

    async def get_operation(self, name: str, access_token: str) -> dict[str, Any]:
        headers = self._headers(access_token)
        last_exc: Exception | None = None
        for base_url in self._base_urls:
            url = f"{base_url}/{name}"
            try:
                response = await self._client.get(url, headers=headers)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as exc:
                if not _should_fallback(exc):
                    raise
                last_exc = exc
            except httpx.HTTPError as exc:
                last_exc = exc
        if last_exc:
            raise last_exc
        raise RuntimeError("Code Assist operation lookup failed without an error response.")

    def _headers(self, access_token: str, model_name: str | None = None) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": _gemini_cli_user_agent(model_name),
        }
        return headers


async def _iter_sse(response: httpx.Response) -> AsyncIterator[dict[str, Any]]:
    buffered: list[str] = []
    async for line in response.aiter_lines():
        if line.startswith("data: "):
            buffered.append(line[6:].strip())
        elif line.strip() == "":
            if not buffered:
                continue
            payload = json.loads("\n".join(buffered))
            buffered = []
            yield payload


async def _parse_rate_limit_error(
    response: httpx.Response,
) -> tuple[float | None, str | None, dict[str, Any] | None]:
    try:
        if not response.is_stream_consumed:
            await response.aread()
        error_data = json.loads(response.content)
        if not isinstance(error_data, dict):
            return None, _fallback_rate_limit_summary(response, "non-object error body"), None
        error_info = error_data.get("error", {})
        if not isinstance(error_info, dict):
            return None, _fallback_rate_limit_summary(response, "missing error object"), None
        details = error_info.get("details", [])
        delay = _extract_retry_delay(details)
        summary = _format_rate_limit_summary(error_info, details)
        return delay, summary, error_info
    except (json.JSONDecodeError, Exception):
        return None, _fallback_rate_limit_summary(response, "unparseable error body"), None


def _parse_duration(duration_str: str) -> float | None:
    if not duration_str or not isinstance(duration_str, str):
        return None
    duration_str = duration_str.strip()
    try:
        if duration_str.endswith("ms"):
            return float(duration_str[:-2]) / 1000.0
        if duration_str.endswith("s"):
            return float(duration_str[:-1])
        return float(duration_str)
    except ValueError:
        return None


def _extract_retry_delay(details: Any) -> float | None:
    if not isinstance(details, list):
        return None
    for detail in details:
        if not isinstance(detail, dict):
            continue
        detail_type = detail.get("@type", "")
        if "RetryInfo" in detail_type:
            retry_delay = detail.get("retryDelay", "")
            parsed = _parse_duration(retry_delay)
            if parsed is not None:
                return parsed
        if "ErrorInfo" in detail_type:
            metadata = detail.get("metadata", {})
            if isinstance(metadata, dict):
                quota_delay = metadata.get("quotaResetDelay", "")
                parsed = _parse_duration(quota_delay)
                if parsed is not None:
                    return parsed
    return None


def _format_rate_limit_summary(error_info: dict[str, Any], details: Any) -> str:
    status = error_info.get("status") or error_info.get("code") or "429"
    message = _truncate(str(error_info.get("message") or ""), _RATE_LIMIT_MESSAGE_MAX)
    summary = f"status={status}"
    if message:
        summary += f" message={message}"
    if isinstance(details, list):
        quota_fields = {}
        for detail in details:
            if not isinstance(detail, dict):
                continue
            if "ErrorInfo" not in str(detail.get("@type", "")):
                continue
            metadata = detail.get("metadata", {})
            if isinstance(metadata, dict):
                for key in ("quotaMetric", "quotaLimit", "quotaLocation", "quotaId", "consumer", "service"):
                    if metadata.get(key):
                        quota_fields[key] = metadata[key]
        if quota_fields:
            summary += f" quota={_truncate(json.dumps(quota_fields, sort_keys=True), _RATE_LIMIT_MESSAGE_MAX)}"
    return summary


def _fallback_rate_limit_summary(response: httpx.Response, reason: str) -> str:
    body = ""
    try:
        body = response.text or ""
    except Exception:
        body = ""
    body = _truncate(body, _RATE_LIMIT_MESSAGE_MAX)
    if body:
        return f"{reason} body={body}"
    return reason


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return f"{value[:limit]}..."


async def _log_http_error(response: httpx.Response, method: str, header_style: str) -> None:
    try:
        if not response.is_stream_consumed:
            await response.aread()
        content_type = response.headers.get("content-type", "")
        body = ""
        if "application/json" in content_type:
            try:
                parsed = response.json()
                body = json.dumps(parsed, sort_keys=True)
            except Exception:
                body = response.text
        else:
            body = response.text
        body = _truncate(body, _RATE_LIMIT_MESSAGE_MAX)
        if body:
            logger.warning(
                "Code Assist HTTP %s on %s (style=%s) body=%s",
                response.status_code,
                method,
                header_style,
                body,
            )
    except Exception:
        return


def _header_style() -> str:
    style_config = os.getenv(CODE_ASSIST_HEADER_STYLES_ENV, DEFAULT_CODE_ASSIST_HEADER_STYLE)
    for item in style_config.split(","):
        key = item.strip().lower()
        if key == "gemini-cli":
            return "gemini-cli"
    return "gemini-cli"


def _header_styles() -> list[str]:
    return [_header_style()]


def _should_abort_rate_limit(error_info: dict[str, Any] | None, delay: float | None) -> bool:
    if not error_info:
        return False
    status = str(error_info.get("status") or "")
    if status != "RESOURCE_EXHAUSTED":
        return False
    if delay is None:
        return True
    return delay >= CODE_ASSIST_RATE_LIMIT_MAX_WAIT_S


def _should_fallback(exc: httpx.HTTPStatusError) -> bool:
    status = exc.response.status_code
    if status == 401:
        return False
    return status in _FALLBACK_STATUSES


def _gemini_cli_user_agent(model_name: str | None) -> str:
    version_text = _isaac_version()
    model_text = model_name or "unknown-model"
    return f"GeminiCLI/{version_text}/{model_text} ({_platform_name()}; {_platform_arch()})"


def _isaac_version() -> str:
    try:
        return version("isaac-acp")
    except PackageNotFoundError:
        return "0.0.0"


def _platform_name() -> str:
    if os.name == "nt":
        return "win32"
    if os.name == "posix" and "darwin" in platform.system().lower():
        return "darwin"
    return "linux"


def _platform_arch() -> str:
    machine = platform.machine().lower()
    if machine in {"x86_64", "amd64"}:
        return "x64"
    if machine in {"aarch64", "arm64"}:
        return "arm64"
    return machine or "unknown"
