"""Process configuration; workspace configuration comes from the ACP client."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from platformdirs import user_config_path, user_state_path
from pydantic_ai_harness.permissions import PermissionMode

PERMISSION_MODES: tuple[PermissionMode, ...] = ("auto", "read-only", "full")


@dataclass
class SessionSettings:
    """Settings editable within one live ACP session, shared with its delegates."""

    permissions: PermissionMode
    planning: bool = False


@dataclass(frozen=True)
class Settings:
    """Options shared by the sessions served by one Isaac process."""

    models: tuple[str, ...]
    permissions: PermissionMode = "auto"
    code_mode: bool = True
    skill_dirs: tuple[str, ...] = (".agents/skills",)
    context_window: int | None = None
    agent_dirs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.models or any(not model.strip() for model in self.models):
            raise ValueError("Set ISAAC_MODEL or pass --model with a Pydantic AI provider:model identifier.")
        if self.context_window is not None and self.context_window <= 0:
            raise ValueError("--context-window must be a positive number of tokens.")


def env_file() -> Path:
    """Use the shared config file regardless of the client's working directory."""
    return user_config_path("isaac", appauthor=False) / ".env"


def load_environment() -> None:
    """Explicit process environment takes precedence over the shared .env."""
    load_dotenv(env_file(), override=False)


def session_database() -> Path:
    """Keep saved sessions outside workspaces, with an optional state-directory override."""
    override = os.environ.get("ISAAC_STATE_DIR")
    directory = Path(override).expanduser() if override else user_state_path("isaac", appauthor=False)
    return directory / "sessions.sqlite3"
