"""Status UI rendering for client components."""

from __future__ import annotations

import os
from pathlib import Path

from prompt_toolkit.utils import get_cwidth  # type: ignore

from isaac.client.session_state import SessionUIState


def format_path(path: str | None) -> str:
    if not path:
        path = os.getcwd()
    resolved = Path(path).expanduser()
    try:
        resolved = resolved.resolve()
    except Exception:
        return str(resolved)
    home = Path.home()
    try:
        rel = resolved.relative_to(home)
    except ValueError:
        return str(resolved)
    return str(Path("~") / rel)


def _format_mcp_summary(mcp_servers: list[str]) -> str:
    if not mcp_servers:
        return "none"
    if len(mcp_servers) == 1:
        return mcp_servers[0]
    return f"{mcp_servers[0]} +{len(mcp_servers) - 1}"


def _format_model_intro(model_id: str | None) -> str:
    if not model_id or model_id == "unknown":
        return "not set, send /setup to configure"
    return model_id


def build_status_toolbar(state: SessionUIState) -> list[tuple[str, str]]:
    mode = state.current_mode or "unknown"
    model = state.current_model or "unknown"
    mcp = _format_mcp_summary(state.mcp_servers)

    usage_text = state.usage_summary
    usage_label = "Usage: "
    if not usage_text:
        usage_text = "n/a"
    elif usage_text.startswith("Usage:"):
        usage_text = usage_text[len("Usage:") :].strip()
    else:
        usage_label = ""

    gap = ("", "  ")
    parts: list[tuple[str, str]] = [
        ("class:toolbar.label", "Mode: "),
        ("class:toolbar.value", mode),
        gap,
        ("class:toolbar.label", "Model: "),
        ("class:toolbar.value", model),
        gap,
        ("class:toolbar.label", "MCP: "),
        ("class:toolbar.value", mcp),
        gap,
    ]
    if usage_label:
        parts.append(("class:toolbar.label", usage_label))
    parts.append(("class:toolbar.value", usage_text))
    parts.extend(
        [
            gap,
            ("class:toolbar.label", "Esc: "),
            ("class:toolbar.value", "cancel"),
            gap,
            ("class:toolbar.label", "Ctrl-Q: "),
            ("class:toolbar.value", "exit"),
        ]
    )
    return parts


def build_welcome_banner(state: SessionUIState) -> str:
    cwd = format_path(state.cwd)
    session_id = state.session_id or "<unknown>"
    model_line = _format_model_intro(state.current_model)

    lines = [
        "🍏 Welcome to Isaac CLI! 🍏",
        "Send /help for help information.",
        "",
        f"Directory: {cwd}",
        f"Session: {session_id}",
        f"Model: {model_line}",
    ]
    content_width = max(_display_width(line) for line in lines)
    padded_lines: list[str] = []
    for idx, line in enumerate(lines):
        if idx in {0, 1}:
            aligned = _center_to_width(line, content_width)
        else:
            aligned = _pad_to_width(line, content_width)
        padded_lines.append(f" {aligned} ")

    width = content_width + 2
    green = "\x1b[32m"
    bold = "\x1b[1m"
    reset = "\x1b[0m"

    top = f"{green}┌{'─' * width}┐{reset}"
    body: list[str] = []
    for idx, line in enumerate(padded_lines):
        content = f"{bold}{line}{reset}" if idx == 0 else line
        body.append(f"{green}│{reset}{content}{green}│{reset}")
    bottom = f"{green}└{'─' * width}┘{reset}"
    return "\n".join([top, *body, bottom, ""])


def _display_width(text: str) -> int:
    return get_cwidth(text)


def _pad_to_width(text: str, width: int) -> str:
    padding = max(0, width - _display_width(text))
    if padding:
        return f"{text}{' ' * padding}"
    return text


def _center_to_width(text: str, width: int) -> str:
    text_width = _display_width(text)
    if text_width >= width:
        return _pad_to_width(text, width)
    padding = width - text_width
    left = padding // 2
    right = padding - left
    return f"{' ' * left}{text}{' ' * right}"
