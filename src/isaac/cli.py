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
    try:
        return dialog.run()
    except RuntimeError:
        # In running event loop, await async dialog.
        return await dialog.run_async()


async def run_cli():
    runner = model_registry.build_agent(
        model_registry.load_models_config().get("current", "test"),
        register_tools,
    )
    current_mode = "ask"
    mode_ids = {m["id"] for m in available_modes()}
    approved_commands: set[str] = set()
    kb = KeyBindings()
    CANCEL_TOKEN = "__CANCEL__"

    @kb.add("escape")
    def _(event):  # type: ignore
        event.app.exit(result=CANCEL_TOKEN)

    session = PromptSession("> ", key_bindings=kb)

    while True:
        try:
            prompt = await session.prompt_async()
            if prompt == CANCEL_TOKEN:
                print("[cancelled]")
                continue
            if prompt.lower() in ["exit", "quit"]:
                break
            if prompt.strip() == "/test":
                print(_run_pytest())
                continue
            if prompt.startswith("/model"):
                parts = prompt.split()
                if len(parts) == 1:
                    models = model_registry.list_models()
                    current = model_registry.load_models_config().get("current", "test")
                    selection = await _select_model_cli(models, current)

                    if selection:
                        try:
                            model_registry.set_current_model(selection)
                            runner = model_registry.build_agent(selection, register_tools)
                            print(f"[switched to model {selection}]")
                        except Exception as exc:
                            print(f"[failed to switch model: {exc}]")
                else:
                    target = parts[1]
                    try:
                        model_registry.set_current_model(target)
                        runner = model_registry.build_agent(target, register_tools)
                        print(f"[switched to model {target}]")
                    except Exception as exc:
                        print(f"[failed to switch model: {exc}]")
                continue

            if prompt.startswith("/mode "):
                requested = prompt[len("/mode ") :].strip()
                if requested in mode_ids:
                    current_mode = requested
                    print(f"[mode set to {current_mode}]")
                else:
                    print(f"[unknown mode: {requested}; available: {', '.join(sorted(mode_ids))}]")
                continue

            if current_mode == "reject":
                print("[request rejected in current mode]")
                continue
            if current_mode == "request_permission":
                if prompt not in approved_commands:
                    result = radiolist_dialog(
                        title="Permission required",
                        text=f"Command: {prompt}",
                        values=[
                            ("y", "Yes, proceed"),
                            ("a", "Yes, and don't ask again for this command"),
                            ("esc", "No, and tell me what to do differently"),
                        ],
                    ).run()

                    if result == "a":
                        approved_commands.add(prompt)
                    elif result not in {"y", "yes"}:
                        print("[request cancelled] If you want me to proceed, allow the action.")
                        continue

            response_text = await run_with_runner(runner, prompt)
            print(response_text)
        except (EOFError, KeyboardInterrupt):
            break
    print("\nExiting simple interactive agent.")
