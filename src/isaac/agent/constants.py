"""Shared constants for the ACP agent."""

# Limit the size of tool/terminal output emitted in a single ACP update.
# Keep updates compact to avoid flooding the client UI or prompt history.
TOOL_OUTPUT_LIMIT = 48 * 1024
