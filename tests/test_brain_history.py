from __future__ import annotations

from isaac.agent.brain.history import build_chat_history
from acp.helpers import session_notification, update_agent_message
from acp.schema import TextContentBlock, UserMessageChunk


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
    ]

    history = build_chat_history(updates)

    assert history == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
