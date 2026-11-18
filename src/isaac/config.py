"""Process configuration; workspace configuration comes from the ACP client."""

from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from platformdirs import user_config_path
from pydantic_ai_harness.permissions import PermissionMode


@dataclass(frozen=True)
class Settings:
    """Options shared by the sessions served by one Isaac process."""

    models: tuple[str, ...]
    permissions: PermissionMode = "auto"
    code_mode: bool = False
    skill_dirs: tuple[str, ...] = (".agents/skills",)
    context_window: int | None = None

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
