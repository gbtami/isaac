from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from acp.agent.connection import AgentSideConnection
from acp import text_block
from acp.schema import AgentMessageChunk

from tests.utils import make_function_agent
from isaac.agent import models as model_registry
from isaac.agent.agent import ACPAgent
from isaac.agent.slash import handle_slash_command


def _raise_model_error(*_: object, **__: object) -> tuple[object, object]:
    raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_set_session_model_changes_runner():
    conn = AsyncMock(spec=AgentSideConnection)
    agent = make_function_agent(conn)
    session = await agent.new_session(cwd="/", mcp_servers=[])

    await agent.set_session_model(model_id="function:function", session_id=session.session_id)

    response = await agent.prompt(prompt=[text_block("hello")], session_id=session.session_id)

    conn.session_update.assert_called()
    notification = conn.session_update.call_args.kwargs["update"]
    assert isinstance(notification, AgentMessageChunk)
    assert notification.content.text
    assert "Error" not in notification.content.text
    assert response.stop_reason == "end_turn"


@pytest.mark.asyncio
async def test_model_build_failure_surfaces_error(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setattr(model_registry, "build_agent_pair", _raise_model_error)

    conn = AsyncMock(spec=AgentSideConnection)
    conn.session_update = AsyncMock()
    agent = ACPAgent(conn)
    session = await agent.new_session(cwd=str(tmp_path), mcp_servers=[])

    updates = [call.kwargs["update"] for call in conn.session_update.await_args_list]  # type: ignore[attr-defined]
    assert any(
        isinstance(u, AgentMessageChunk)
        and "Model load failed" in getattr(getattr(u, "content", None), "text", "")
        for u in updates
    )

    response = await agent.prompt(prompt=[text_block("hi")], session_id=session.session_id)
    assert response.stop_reason == "refusal"


@pytest.mark.asyncio
async def test_unknown_model_id_is_rejected(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    conn = AsyncMock(spec=AgentSideConnection)
    conn.session_update = AsyncMock()
    agent = make_function_agent(conn)
    session = await agent.new_session(cwd=str(tmp_path), mcp_servers=[])

    slash = await handle_slash_command(agent, session.session_id, "/model no-such")
    assert slash is not None
    await agent._send_update(slash)  # type: ignore[attr-defined]

    updates = [call.kwargs["update"] for call in conn.session_update.await_args_list]  # type: ignore[attr-defined]
    assert any(
        isinstance(u, AgentMessageChunk)
        and "Unknown model id: no-such" in getattr(getattr(u, "content", None), "text", "")
        for u in updates
    )
