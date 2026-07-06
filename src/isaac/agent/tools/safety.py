"""Shared safety helpers for Isaac's filesystem and shell tools.

These helpers intentionally keep Isaac's public ACP tool names stable while
borrowing the most useful hardening ideas from Pydantic AI Harness: workspace
containment, symlink-aware path checks, protected write patterns, binary-file
refusal for text tools, file hashes, and configurable shell allow/deny checks.
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
import re
from pathlib import Path
from typing import Iterable

DEFAULT_IGNORES = {
    ".git",
    ".hg",
    ".svn",
    ".cache",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".tox",
    ".nox",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".uv-cache",
    "node_modules",
}

# Write protection is deliberately narrower than read/list protection. These
# paths are easy for an agent to corrupt accidentally and are rarely legitimate
# targets for autonomous file replacement.
PROTECTED_WRITE_GLOBS = (
    ".git",
    ".git/**",
    "**/.git",
    "**/.git/**",
    ".env",
    ".env.*",
    "**/.env",
    "**/.env.*",
    "*.pem",
    "*.key",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "**/id_rsa",
    "**/id_dsa",
    "**/id_ecdsa",
    "**/id_ed25519",
)

# Built-in catastrophic command blockers. This is not a sandbox; it only catches
# common foot-guns before they reach approval/yolo execution.
CATASTROPHIC_SHELL_DENY_PATTERNS = (
    r":\s*\(\)\s*\{\s*:\|:\s*&\s*\}\s*;\s*:",  # classic fork bomb
    r"\brm\s+(?:-[^\n\r\s]*[rR][^\n\r\s]*[fF]|-[^\n\r\s]*[fF][^\n\r\s]*[rR])\s+(?:/|~|\$HOME)(?:\s|$)",
    r"\bmkfs(?:\.[\w-]+)?\b",
    r"\bdd\b[^\n\r]*(?:\s|^)of=/dev/",
)

_BINARY_SAMPLE_BYTES = 8192
_MAX_SHELL_COMMAND_CHARS = 20_000


class PathAccessError(ValueError):
    """Raised when a tool path violates workspace policy."""


class ProtectedPathError(ValueError):
    """Raised when a mutating tool targets a protected path."""


class BinaryFileError(ValueError):
    """Raised when a text tool is asked to process binary content."""


class ShellCommandDenied(ValueError):
    """Raised when a shell command violates configured policy."""


def _coerce_base(base: str | Path | None) -> Path | None:
    if base is None:
        return None
    return Path(base).expanduser().resolve(strict=False)


def _coerce_roots(roots: Iterable[str | Path] | None) -> tuple[Path, ...]:
    if roots is None:
        return ()
    coerced: list[Path] = []
    for root in roots:
        try:
            coerced.append(Path(root).expanduser().resolve(strict=False))
        except (OSError, RuntimeError, TypeError, ValueError):
            continue
    return tuple(coerced)


def is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
    except ValueError:
        return False
    return True


def resolve_workspace_path(
    base: str | Path | None,
    target: str | Path,
    *,
    allow_outside: bool = False,
    additional_directories: Iterable[str | Path] | None = None,
) -> Path:
    """Resolve ``target`` with symlink-aware optional workspace containment.

    When ``base`` is provided, relative targets are resolved under it and both
    relative and absolute targets must remain under ``base`` unless
    ``allow_outside`` is explicitly true. ``Path.resolve(strict=False)`` follows
    existing symlinks in parents, so symlink escapes are caught even for files
    that do not exist yet.
    """

    base_path = _coerce_base(base)
    raw = Path(target).expanduser()
    candidate = raw if raw.is_absolute() else (base_path or Path.cwd().resolve(strict=False)) / raw
    resolved = candidate.resolve(strict=False)
    if base_path is not None and not allow_outside:
        allowed_roots = (base_path, *_coerce_roots(additional_directories))
        if not any(is_relative_to(resolved, root) for root in allowed_roots):
            raise PathAccessError("Path is outside allowed working directory")
    return resolved


def display_path(path: Path, base: str | Path | None = None) -> str:
    """Return a compact display path for tool output."""

    base_path = _coerce_base(base)
    if base_path is not None:
        try:
            return path.relative_to(base_path).as_posix()
        except ValueError:
            pass
    return path.as_posix()


def path_matches_any(path: Path | str, patterns: Iterable[str]) -> bool:
    rel = path.as_posix() if isinstance(path, Path) else path.replace(os.sep, "/")
    rel = rel.lstrip("/")
    name = Path(rel).name
    return any(fnmatch.fnmatchcase(rel, pattern) or fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


def is_default_ignored(path: Path) -> bool:
    return any(part in DEFAULT_IGNORES for part in path.parts)


def ensure_not_protected_for_write(path: Path, base: str | Path | None = None) -> None:
    rel = Path(display_path(path, base))
    if path_matches_any(rel, PROTECTED_WRITE_GLOBS):
        raise ProtectedPathError(f"Refusing to write protected path: {rel.as_posix()}")


def is_binary_file(path: Path) -> bool:
    """Return true when a file looks unsafe for UTF-8 text tools."""

    try:
        sample = path.read_bytes()[:_BINARY_SAMPLE_BYTES]
    except OSError:
        return False
    if not sample:
        return False
    if b"\x00" in sample:
        return True
    try:
        sample.decode("utf-8")
    except UnicodeDecodeError:
        return True
    return False


def ensure_text_file(path: Path) -> None:
    if path.exists() and path.is_file() and is_binary_file(path):
        raise BinaryFileError(f"Refusing to read binary file: {path.name}")


def ensure_text_target(path: Path, base: str | Path | None = None) -> None:
    ensure_not_protected_for_write(path, base)
    if path.exists() and path.is_file() and is_binary_file(path):
        raise BinaryFileError(f"Refusing to overwrite binary file: {path.name}")


def sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _split_env_patterns(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for part in value.splitlines() for item in part.split(",") if item.strip()]


def _regex_matches_any(command: str, patterns: Iterable[str]) -> str | None:
    for pattern in patterns:
        try:
            if re.search(pattern, command):
                return pattern
        except re.error:
            # Treat malformed user-supplied policy as non-matching instead of
            # breaking every command invocation.
            continue
    return None


def validate_shell_command(command: str) -> None:
    """Apply lightweight, configurable shell policy.

    ``ISAAC_SHELL_ALLOWLIST`` and ``ISAAC_SHELL_DENYLIST`` accept comma- or
    newline-separated regular expressions. An allowlist, when present, means at
    least one pattern must match. Denylists always win. Built-in catastrophic
    blockers run regardless of env configuration.
    """

    normalized = command.strip()
    if not normalized:
        raise ShellCommandDenied("Command is empty")
    if len(normalized) > _MAX_SHELL_COMMAND_CHARS:
        raise ShellCommandDenied(f"Command is too long ({len(normalized)} characters)")

    catastrophic = _regex_matches_any(normalized, CATASTROPHIC_SHELL_DENY_PATTERNS)
    if catastrophic is not None:
        raise ShellCommandDenied("Command matches built-in catastrophic deny pattern")

    custom_deny = _regex_matches_any(normalized, _split_env_patterns(os.getenv("ISAAC_SHELL_DENYLIST")))
    if custom_deny is not None:
        raise ShellCommandDenied(f"Command denied by ISAAC_SHELL_DENYLIST pattern: {custom_deny}")

    allowlist = _split_env_patterns(os.getenv("ISAAC_SHELL_ALLOWLIST"))
    if allowlist and _regex_matches_any(normalized, allowlist) is None:
        raise ShellCommandDenied("Command does not match ISAAC_SHELL_ALLOWLIST")


def resolve_command_cwd(
    session_cwd: str | Path | None,
    cwd: str | Path | None,
    *,
    additional_directories: Iterable[str | Path] | None = None,
) -> Path:
    resolved = resolve_workspace_path(session_cwd, cwd or ".", additional_directories=additional_directories)
    if not resolved.exists():
        raise PathAccessError(f"Working directory does not exist: {cwd or '.'}")
    if not resolved.is_dir():
        raise PathAccessError(f"Working directory is not a directory: {cwd or '.'}")
    return resolved
