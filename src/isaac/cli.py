"""Launch the ACP stdio server; the ACP client supplies the user interface."""

import argparse
import asyncio
import logging
import os
import sys
import warnings

from isaac import __version__
from isaac.config import PERMISSION_MODES, Settings, load_environment


def parse_settings(argv: list[str] | None = None) -> Settings:
    """Read CLI options after loading the shared provider environment."""
    load_environment()
    parser = argparse.ArgumentParser(
        description="Serve the Isaac Harness coding agent over ACP stdio.",
        epilog="Run `isaac login` to sign in with your ChatGPT/Codex subscription.",
    )
    parser.add_argument("--version", action="version", version=f"isaac {__version__}")
    parser.add_argument(
        "--model",
        action="append",
        help="Pydantic AI provider:model ID; repeat for the ACP model menu. Defaults to ISAAC_MODEL.",
    )
    parser.add_argument(
        "--permissions",
        choices=PERMISSION_MODES,
        default="auto",
        help="Harness permission mode (default: auto).",
    )
    parser.add_argument(
        "--code-mode",
        action=argparse.BooleanOptionalAction,
        default=Settings.code_mode,
        help="Expose eligible tools through Harness's Monty Code Mode (default: enabled).",
    )
    parser.add_argument(
        "--skill-dir",
        action="append",
        help="Skill library in each session's workspace; repeat as needed (default: .agents/skills).",
    )
    parser.add_argument(
        "--agent-dir", action="append", help="Folder of named delegate definitions in each workspace. Repeat as needed."
    )
    parser.add_argument(
        "--context-window", type=int, help="Override the model context window in tokens for compaction."
    )
    args = parser.parse_args(argv)
    models = args.model or [os.environ.get("ISAAC_MODEL", "")]
    try:
        return Settings(
            models=tuple(dict.fromkeys(model.strip() for model in models)),
            permissions=args.permissions,
            code_mode=args.code_mode,
            skill_dirs=tuple(args.skill_dir) if args.skill_dir is not None else (".agents/skills",),
            context_window=args.context_window,
            agent_dirs=tuple(args.agent_dir or ()),
        )
    except ValueError as error:
        parser.error(str(error))


def main(argv: list[str] | None = None) -> None:
    """Reserve stdout for ACP JSON-RPC and send diagnostics to stderr."""
    argv = list(argv) if argv is not None else sys.argv[1:]
    if argv and argv[0] == "login":
        parser = argparse.ArgumentParser(prog="isaac login", description="Sign in to ChatGPT/Codex in a browser.")
        parser.add_argument("--no-browser", action="store_true", help="Print the login URL without opening a browser.")
        args = parser.parse_args(argv[1:])
        from pydantic_ai.exceptions import UserError

        from isaac.auth import login

        load_environment()
        try:
            asyncio.run(login(open_browser=not args.no_browser))
        except (UserError, OSError) as error:
            print(f"isaac login: {error}", file=sys.stderr)
            raise SystemExit(1) from None
        except KeyboardInterrupt:
            raise SystemExit(130) from None
        return
    settings = parse_settings(argv)
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)

    # Harness ACP is intentionally experimental; suppress only its import-time notice.
    # Other Harness experimental warnings remain visible.
    from pydantic_ai_harness.experimental import HarnessExperimentalWarning

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"^`pydantic_ai_harness\.experimental\.acp` is experimental",
            category=HarnessExperimentalWarning,
        )
        from isaac.server import serve

    try:
        asyncio.run(serve(settings))
    except KeyboardInterrupt:
        pass
