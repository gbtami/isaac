"""Shared app directory helpers based on platformdirs."""

from __future__ import annotations

from pathlib import Path

from platformdirs import PlatformDirs

APP_NAME = "isaac"


def _platform_dirs() -> PlatformDirs:
    return PlatformDirs(appname=APP_NAME, appauthor=False)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def config_dir() -> Path:
    return ensure_dir(Path(_platform_dirs().user_config_path))


def state_dir() -> Path:
    return ensure_dir(Path(_platform_dirs().user_state_path))


def cache_dir() -> Path:
    return ensure_dir(Path(_platform_dirs().user_cache_path))


def log_dir() -> Path:
    return ensure_dir(Path(_platform_dirs().user_log_path))
