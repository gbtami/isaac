"""ACP runtime configuration shared by agent and client entrypoints."""

from __future__ import annotations

import os

DEFAULT_STDIO_BUFFER_LIMIT_BYTES = 50 * 1024 * 1024
_MIN_STDIO_BUFFER_LIMIT_BYTES = 64 * 1024


def _parse_stdio_buffer_limit(raw_value: str | None) -> int:
    if raw_value is None:
        return DEFAULT_STDIO_BUFFER_LIMIT_BYTES
    try:
        parsed = int(raw_value)
    except ValueError:
        return DEFAULT_STDIO_BUFFER_LIMIT_BYTES
    return max(parsed, _MIN_STDIO_BUFFER_LIMIT_BYTES)


ACP_STDIO_BUFFER_LIMIT_BYTES = _parse_stdio_buffer_limit(os.getenv("ISAAC_ACP_STDIO_BUFFER_LIMIT_BYTES"))
