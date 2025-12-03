"""Status box rendering for client UI."""

from __future__ import annotations

from isaac.client.session_state import SessionUIState


def render_status_box(state: SessionUIState) -> str:
    lines: list[str] = []
    lines.append(f"Mode: {state.current_mode or 'unknown'}")
    lines.append(f"Model: {state.current_model or 'unknown'}")
    if state.mcp_servers:
        for srv in state.mcp_servers:
            lines.append(f"MCP: {srv}")
    else:
        lines.append("MCP: <none>")
    if state.usage_summary:
        lines.append(f"Context: {state.usage_summary}")

    width = max(len(line) for line in lines + ["Status"]) + 2
    border = "+" + "=" * width + "+"
    header = f"| {'Status':^{width}} |"
    body = [f"| {line:<{width}} |" for line in lines]
    return "\n".join([border, header, border, *body, border])
