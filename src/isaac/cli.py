"""Launch the ACP stdio server; the ACP client supplies the user interface."""

import argparse
import asyncio
import logging
import os
import sys

from isaac import __version__
from isaac.config import Settings, load_environment


def parse_settings(argv: list[str] | None = None) -> Settings:
    """Read CLI options after loading the shared provider environment."""
    load_environment()
    parser = argparse.ArgumentParser(description="Serve the Isaac Harness coding agent over ACP stdio.")
    parser.add_argument("--version", action="version", version=f"isaac {__version__}")
    parser.add_argument(
        "--model",
        action="append",
        help="Pydantic AI provider:model ID; repeat for the ACP model menu. Defaults to ISAAC_MODEL.",
    )
    parser.add_argument(
        "--permissions",
        choices=("auto", "read-only", "full"),
        default="auto",
        help="Harness permission mode (default: auto).",
    )
    parser.add_argument("--code-mode", action="store_true", help="Expose tools through Harness's Monty CodeMode.")
    parser.add_argument(
        "--skill-dir",
        action="append",
        help="Skill library in each session's workspace; repeat as needed (default: .agents/skills).",
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
        )
    except ValueError as error:
        parser.error(str(error))


def main(argv: list[str] | None = None) -> None:
    """Reserve stdout for ACP JSON-RPC and send diagnostics to stderr."""
    settings = parse_settings(argv)
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)

    from isaac.server import serve

    try:
        asyncio.run(serve(settings))
    except KeyboardInterrupt:
        pass
