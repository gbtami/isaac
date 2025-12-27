"""HTTP fetch tool for pulling external docs/content."""

from __future__ import annotations

import ipaddress
import os
import time
from collections import deque
from typing import Any, Dict
from urllib.parse import urlparse

import httpx
from aiocache import SimpleMemoryCache


SAFE_SCHEMES = {"https"}

DEFAULT_FETCH_MAX_BYTES = 20_000
DEFAULT_FETCH_TIMEOUT = 30.0
DEFAULT_FETCH_WINDOW_S = 10.0
DEFAULT_FETCH_MAX_CALLS = 6
DEFAULT_FETCH_ERROR_THRESHOLD = 3
DEFAULT_FETCH_COOLDOWN_S = 30.0
FETCH_CACHE_MAX_SIZE = 32
# aiocache.SimpleMemoryCache in this version does not accept maxsize; keep default size.
_CACHE = SimpleMemoryCache()

_FETCH_CALLS: dict[str, deque[float]] = {}
_FETCH_ERRORS: dict[str, int] = {}
_FETCH_COOLDOWNS: dict[str, float] = {}


def _fetch_limit_config() -> tuple[float, int, int, float]:
    """Load rate-limit and cooldown settings from the environment."""
    window_s = float(os.getenv("ISAAC_FETCH_WINDOW_S", DEFAULT_FETCH_WINDOW_S))
    max_calls = int(os.getenv("ISAAC_FETCH_MAX_CALLS", DEFAULT_FETCH_MAX_CALLS))
    error_threshold = int(os.getenv("ISAAC_FETCH_ERROR_THRESHOLD", DEFAULT_FETCH_ERROR_THRESHOLD))
    cooldown_s = float(os.getenv("ISAAC_FETCH_COOLDOWN_S", DEFAULT_FETCH_COOLDOWN_S))
    return window_s, max_calls, error_threshold, cooldown_s


def _rate_limit_host(host: str) -> str | None:
    """Apply a sliding-window limit per host to avoid runaway fetches.

    Returns an error message when the host is throttled; otherwise None.
    """
    window_s, max_calls, _, cooldown_s = _fetch_limit_config()
    now = time.monotonic()
    cooldown_until = _FETCH_COOLDOWNS.get(host)
    if cooldown_until and now < cooldown_until:
        return f"fetch_url cooldown active for host {host}"

    calls = _FETCH_CALLS.setdefault(host, deque())
    while calls and now - calls[0] > window_s:
        calls.popleft()
    if len(calls) >= max_calls:
        _FETCH_COOLDOWNS[host] = now + cooldown_s
        return f"fetch_url rate limit exceeded for host {host}"

    calls.append(now)
    return None


def _record_fetch_error(host: str) -> None:
    """Track repeated failures and trigger cooldowns after too many errors."""
    _, _, error_threshold, cooldown_s = _fetch_limit_config()
    errors = _FETCH_ERRORS.get(host, 0) + 1
    _FETCH_ERRORS[host] = errors
    if errors >= error_threshold:
        _FETCH_COOLDOWNS[host] = time.monotonic() + cooldown_s


def _record_fetch_success(host: str) -> None:
    """Reset the error counter after a successful fetch."""
    _FETCH_ERRORS.pop(host, None)


def _blocked_host(host: str | None) -> bool:
    """Reject local or private hosts to avoid SSRF-style access."""
    if not host:
        return True
    lowered = host.lower()
    if lowered in {"localhost"}:
        return True
    try:
        ip_obj = ipaddress.ip_address(lowered)
        if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local:
            return True
    except ValueError:
        if lowered.endswith(".local"):
            return True
    return False


async def fetch_url(
    url: str, max_bytes: int = DEFAULT_FETCH_MAX_BYTES, timeout: float | None = DEFAULT_FETCH_TIMEOUT
) -> Dict[str, Any]:
    """Fetch a URL with basic safety guards and size limits.

    The helper enforces https-only access, blocks local/private targets, caches
    successes, and rate limits per host to reduce the chance of escalations or
    provider throttling from repeated retries.
    """

    parsed = urlparse(url)
    if parsed.scheme not in SAFE_SCHEMES:
        return {"content": None, "error": "Only https URLs are allowed."}
    if _blocked_host(parsed.hostname):
        return {"content": None, "error": "Blocked host for security reasons."}
    if max_bytes <= 0:
        return {"content": None, "error": "max_bytes must be positive."}

    cache_key = (url, max_bytes, timeout)
    cached = await _CACHE.get(cache_key)
    if cached is not None:
        cached_copy = dict(cached)
        cached_copy["cached"] = True
        return cached_copy

    host = parsed.hostname or ""
    if host:
        rate_error = _rate_limit_host(host)
        if rate_error:
            return {"content": None, "error": rate_error}

    headers = {
        "User-Agent": "isaac-fetch/1.0",
        "Accept": "text/*, application/json;q=0.9, */*;q=0.1",
    }
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
            async with client.stream("GET", url, headers=headers) as response:
                content_type = response.headers.get("content-type", "")
                final_url = str(response.url)
                status_code = response.status_code
                truncated = False
                collected = bytearray()
                async for chunk in response.aiter_bytes():
                    if len(collected) + len(chunk) > max_bytes:
                        keep = max_bytes - len(collected)
                        if keep > 0:
                            collected.extend(chunk[:keep])
                        truncated = True
                        break
                    collected.extend(chunk)
                text = collected.decode(response.encoding or "utf-8", errors="replace")
                error = None
                if status_code >= 400:
                    error = f"HTTP {status_code} for {final_url}"
                result = {
                    "content": text,
                    "error": error,
                    "status_code": status_code,
                    "url": final_url,
                    "content_type": content_type,
                    "truncated": truncated,
                }
                if error:
                    if host:
                        _record_fetch_error(host)
                else:
                    if host:
                        _record_fetch_success(host)
                    await _CACHE.set(cache_key, result)
                return result
    except Exception as exc:  # pragma: no cover - network failures
        if host:
            _record_fetch_error(host)
        return {"content": None, "error": str(exc)}


async def _clear_fetch_cache() -> None:
    """Clear the in-memory fetch cache (used in tests)."""

    await _CACHE.clear()
