from __future__ import annotations

from typing import Any, Dict

from prompt_toolkit import PromptSession  # type: ignore
from prompt_toolkit.key_binding import KeyBindings  # type: ignore
from prompt_toolkit.shortcuts import radiolist_dialog  # type: ignore

from isaac import models as model_registry
from isaac.runner import register_tools, run_with_runner
from isaac.session_modes import available_modes
from isaac.slash import _run_pytest


async def _select_model_cli(
    models: Dict[str, Any], current: str, selection_fallback: str | None = None
) -> str | None:
    if selection_fallback is not None:
        return selection_fallback

    dialog = radiolist_dialog(
        title="Select model",
        text=f"Current: {current}",
        values=[(mid, f"{mid} ({meta.get('description','')})") for mid, meta in models.items()],
    )
    return await dialog.run_async()


def _handle_mode_cli(current_mode: str, mode_ids: set[str], prompt: str) -> tuple[bool, str, str]:
    if prompt.strip() == "/mode":
        available = ", ".join(sorted(mode_ids))
        return True, current_mode, f"Current mode: {current_mode}. Available: {available}"

    if prompt.startswith("/mode "):
        requested = prompt[len("/mode ") :].strip()
        if requested in mode_ids:
            return True, requested, f"[mode set to {requested}]"
        return True, current_mode, f"[unknown mode: {requested}; available: {', '.join(sorted(mode_ids))}]"

    return False, current_mode, ""


async def run_cli():
    """CLI logic now lives in client.py interactive loop."""
    raise NotImplementedError("Use isaac.client interactive loop instead.")
