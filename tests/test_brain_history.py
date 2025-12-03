from __future__ import annotations

from isaac.agent.brain.history import build_chat_history
from acp.helpers import session_notification, update_agent_message, tool_content
from acp.schema import TextContentBlock, ToolCallProgress, UserMessageChunk


def test_build_chat_history_from_updates():
    updates = [
        session_notification(
            "s1",
            UserMessageChunk(
                sessionUpdate="user_message_chunk", content=TextContentBlock(type="text", text="hi")
            ),
        ),
        session_notification(
            "s1",
            update_agent_message(TextContentBlock(type="text", text="hello")),
        ),
        session_notification(
            "s1",
            ToolCallProgress(
                sessionUpdate="tool_call_update",
                status="completed",
                toolCallId="tc1",
                rawOutput={"tool": "tool_run_command", "content": "done"},
                content=[tool_content(TextContentBlock(type="text", text="tool output"))],
            ),
        ),
    ]

    history = build_chat_history(updates)

    assert history == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "assistant", "content": "tool output"},
    ]
