"""HTTP fetch tool for pulling external docs/content."""

from __future__ import annotations

import ipaddress
from typing import Any, Dict
from urllib.parse import urlparse

import httpx
from aiocache import SimpleMemoryCache


SAFE_SCHEMES = {"https"}
_CACHE = SimpleMemoryCache(maxsize=32)


def _blocked_host(host: str | None) -> bool:
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


async def fetch_url(url: str, max_bytes: int = 20_000, timeout: float | None = 10.0) -> Dict[str, Any]:
    """Fetch a URL with basic safety guards and size limits."""

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
                result = {
                    "content": text,
                    "error": None,
                    "status_code": status_code,
                    "url": final_url,
                    "content_type": content_type,
                    "truncated": truncated,
                }
                await _CACHE.set(cache_key, result)
                return result
    except Exception as exc:  # pragma: no cover - network failures
        return {"content": None, "error": str(exc)}


async def _clear_fetch_cache() -> None:
    """Clear the in-memory fetch cache (used in tests)."""

    await _CACHE.clear()
