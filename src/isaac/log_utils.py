"""Logging configuration and structured context helpers."""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import os
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Iterator

from isaac.paths import log_dir

DEFAULT_LOG_DIR = log_dir()
DEFAULT_LOG_MAX_BYTES = 5_000_000
DEFAULT_LOG_BACKUPS = 3

_LOG_CONTEXT: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar("isaac_log_context", default={})
_LOG_CHUNKS_ENABLED = False


@dataclass(frozen=True)
class LogConfig:
    """Configuration for log setup.

    This keeps logging behavior consistent across agent/client entrypoints and
    makes log settings easy to override via environment variables.
    """

    log_file: Path
    level: int = logging.INFO
    stderr: bool = False
    json: bool = False
    log_chunks: bool = False
    max_bytes: int = DEFAULT_LOG_MAX_BYTES
    backup_count: int = DEFAULT_LOG_BACKUPS
    logger_levels: Dict[str, int] = field(default_factory=dict)


def _parse_level(value: str | None, default: int) -> int:
    """Parse a log level string or numeric value from environment settings."""
    if not value:
        return default
    if value.isdigit():
        return int(value)
    return logging._nameToLevel.get(value.upper(), default)


def _parse_bool(value: str | None, default: bool) -> bool:
    """Parse a truthy/falsy toggle from environment settings."""
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_int(value: str | None, default: int) -> int:
    """Parse an integer setting, returning the default on invalid input."""
    if value is None:
        return default
    with contextlib.suppress(ValueError):
        return int(value)
    return default


def build_log_config(*, log_file_name: str, default_level: int = logging.INFO) -> LogConfig:
    """Build log configuration from environment defaults.

    This centralizes log configuration to keep the agent and client aligned on
    file locations, JSON formatting, and chunk logging behavior.
    """

    log_dir = Path(os.getenv("ISAAC_LOG_DIR", str(DEFAULT_LOG_DIR)))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / log_file_name

    return LogConfig(
        log_file=log_file,
        level=_parse_level(os.getenv("ISAAC_LOG_LEVEL"), default_level),
        stderr=_parse_bool(os.getenv("ISAAC_LOG_STDERR"), False),
        json=_parse_bool(os.getenv("ISAAC_LOG_JSON"), False),
        log_chunks=_parse_bool(os.getenv("ISAAC_LOG_CHUNKS"), False),
        max_bytes=_parse_int(os.getenv("ISAAC_LOG_MAX_BYTES"), DEFAULT_LOG_MAX_BYTES),
        backup_count=_parse_int(os.getenv("ISAAC_LOG_BACKUPS"), DEFAULT_LOG_BACKUPS),
    )


def configure_logging(config: LogConfig) -> None:
    """Configure root logging with rotation and context support.

    We reset root handlers to avoid duplicate logs across reloads, then attach
    a file handler (and optional stderr) that adds structured fields.
    """

    global _LOG_CHUNKS_ENABLED
    _LOG_CHUNKS_ENABLED = config.log_chunks

    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    root_logger.setLevel(config.level)

    formatter: logging.Formatter
    if config.json:
        formatter = JsonFormatter()
    else:
        formatter = ContextFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    file_handler = RotatingFileHandler(
        config.log_file,
        maxBytes=config.max_bytes,
        backupCount=config.backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.addFilter(ContextFilter())
    root_logger.addHandler(file_handler)

    if config.stderr:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.addFilter(ContextFilter())
        root_logger.addHandler(stream_handler)

    for name, level in config.logger_levels.items():
        logging.getLogger(name).setLevel(level)


def log_chunks_enabled() -> bool:
    """Return True if per-chunk logging is enabled.

    Chunk logging is noisy by default; this gate allows opt-in debugging only.
    """

    return _LOG_CHUNKS_ENABLED


@contextlib.contextmanager
def log_context(**fields: Any) -> Iterator[None]:
    """Attach structured context fields to log records within a block.

    Context fields avoid repeating session/tool identifiers on every log call,
    while still keeping them available for downstream filtering.
    """

    current = _LOG_CONTEXT.get()
    merged = {**current, **{k: v for k, v in fields.items() if v is not None}}
    token = _LOG_CONTEXT.set(merged)
    try:
        yield
    finally:
        _LOG_CONTEXT.reset(token)


def log_event(logger: logging.Logger, event: str, *, level: int = logging.INFO, **fields: Any) -> None:
    """Log a structured event with optional fields.

    Events are short, stable identifiers that make log search and aggregation
    easier than arbitrary free-form messages.
    """

    logger.log(level, event, extra={"event_fields": fields})


def _format_value(value: Any) -> str:
    """Format a field value for the text formatter."""
    if isinstance(value, str):
        if value == "":
            return '""'
        if any(ch.isspace() for ch in value) or "=" in value or '"' in value:
            return json.dumps(value)
        return value
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    return str(value)


def _format_fields(fields: Dict[str, Any]) -> str:
    """Render field dicts into key=value strings for log lines."""
    parts: list[str] = []
    for key in sorted(fields):
        value = fields.get(key)
        if value is None:
            continue
        parts.append(f"{key}={_format_value(value)}")
    return " ".join(parts)


class ContextFilter(logging.Filter):
    """Inject context fields into log records.

    This keeps structured context alongside log records without manual plumbing
    at each call site.
    """

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003 - logging API name
        record.context_fields = dict(_LOG_CONTEXT.get())
        record.event_fields = getattr(record, "event_fields", {})
        return True


class ContextFormatter(logging.Formatter):
    """Format records with appended structured fields.

    This preserves human-readable logs while still surfacing context fields.
    """

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        context = _format_fields(getattr(record, "context_fields", {}))
        event_fields = _format_fields(getattr(record, "event_fields", {}))
        extra = " ".join(part for part in (context, event_fields) if part)
        if extra:
            return f"{base} {extra}"
        return base


class JsonFormatter(logging.Formatter):
    """Emit JSON log lines with context fields.

    JSON logs help when shipping to log aggregation systems or using jq.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        context = getattr(record, "context_fields", {})
        fields = getattr(record, "event_fields", {})
        if context:
            payload["context"] = context
        if fields:
            payload["fields"] = fields
        return json.dumps(payload, ensure_ascii=True)
