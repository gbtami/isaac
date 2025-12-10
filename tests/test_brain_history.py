from __future__ import annotations

from isaac.agent.brain.history import build_chat_history
from acp import text_block
from acp.helpers import session_notification, update_agent_message, tool_content
from acp.schema import ToolCallProgress, UserMessageChunk


def test_build_chat_history_from_updates():
    updates = [
        session_notification(
            "s1",
            UserMessageChunk(
                session_update="user_message_chunk",
                content=text_block("hi"),
            ),
        ),
        session_notification(
            "s1",
            update_agent_message(text_block("hello")),
        ),
        session_notification(
            "s1",
            ToolCallProgress(
                session_update="tool_call_update",
                status="completed",
                tool_call_id="tc1",
                raw_output={"tool": "run_command", "content": "done"},
                content=[tool_content(text_block("tool output"))],
            ),
        ),
    ]

    history = build_chat_history(updates)

    assert history == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello\ntool output"},
    ]


def test_build_chat_history_merges_streamed_chunks():
    updates = [
        session_notification(
            "s1",
            UserMessageChunk(
                session_update="user_message_chunk",
                content=text_block("hi"),
            ),
        ),
        session_notification("s1", update_agent_message(text_block("Hel"))),
        session_notification("s1", update_agent_message(text_block("lo"))),
        session_notification("s1", update_agent_message(text_block(", world!"))),
    ]

    history = build_chat_history(updates)

    assert history == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello, world!"},
    ]


def test_build_chat_history_skips_slash_commands():
    updates = [
        session_notification(
            "s1",
            UserMessageChunk(
                session_update="user_message_chunk",
                content=text_block("/strategy single"),
            ),
        ),
        session_notification(
            "s1",
            update_agent_message(text_block("Prompt strategy set to single.")),
        ),
        session_notification(
            "s1",
            UserMessageChunk(
                session_update="user_message_chunk",
                content=text_block("next prompt"),
            ),
        ),
        session_notification(
            "s1",
            update_agent_message(text_block("assistant reply")),
        ),
    ]

    history = build_chat_history(updates)

    assert history == [
        {"role": "user", "content": "next prompt"},
        {"role": "assistant", "content": "assistant reply"},
    ]
