"""Shared constants for the ACP agent."""

# Limit the size of tool/terminal output emitted in a single ACP update.
# Using 48KB (vs. the 64KB default StreamReader line limit) leaves room for
# JSON framing and protocol metadata so we avoid LimitOverrun errors on large
# tool responses.
TOOL_OUTPUT_LIMIT = 48 * 1024
