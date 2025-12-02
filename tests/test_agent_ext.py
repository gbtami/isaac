from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from acp import (
    AgentSideConnection,
    InitializeRequest,
    LoadSessionRequest,
    NewSessionRequest,
    PROTOCOL_VERSION,
)
from acp.helpers import session_notification, update_agent_message
from acp.schema import AgentMessageChunk, SessionNotification, TextContentBlock, UserMessageChunk

from isaac.agent.agent import ACPAgent
from isaac.agent import models as model_registry


def _make_user_chunk(session_id: str, text: str) -> SessionNotification:
    return SessionNotification(
        sessionId=session_id,
        update=UserMessageChunk(
            sessionUpdate="user_message_chunk", content=TextContentBlock(type="text", text=text)
        ),
    )


def _make_agent_chunk(session_id: str, text: str) -> SessionNotification:
    return session_notification(
        session_id,
        update_agent_message(TextContentBlock(type="text", text=text)),
    )


@pytest.mark.asyncio
async def test_initialize_advertises_ext_methods(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("HOME", str(tmp_path))
    conn = AsyncMock(spec=AgentSideConnection)
    agent = ACPAgent(conn)

    resp = await agent.initialize(InitializeRequest(protocolVersion=PROTOCOL_VERSION))

    meta = getattr(resp.agentCapabilities, "field_meta", {}) or {}
    assert "extMethods" in meta
    assert {"model/list", "model/set"}.issubset(set(meta["extMethods"]))


@pytest.mark.asyncio
async def test_ext_methods_list_and_set(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(
        model_registry,
        "MODELS_FILE",
        tmp_path / "xdg" / "isaac" / "models.json",
    )
    minimal_config = {
        "current": "function-model",
        "models": {
            "function-model": {
                "provider": "function",
                "model": "function",
                "description": "In-process function model for deterministic testing",
            },
            "user-function": {
                "provider": "function",
                "model": "function",
                "description": "User-visible function model for tests",
            },
        },
    }
    monkeypatch.setattr(model_registry, "DEFAULT_CONFIG", minimal_config)
    model_registry.MODELS_FILE.parent.mkdir(parents=True, exist_ok=True)
    model_registry.MODELS_FILE.write_text(json.dumps(minimal_config), encoding="utf-8")
    conn = AsyncMock(spec=AgentSideConnection)
    agent = ACPAgent(conn)
    session = await agent.newSession(NewSessionRequest(cwd=str(tmp_path), mcpServers=[]))

    listing = await agent.extMethod("model/list", {"sessionId": session.sessionId})
    assert listing.get("current")
    models = listing.get("models", [])
    assert isinstance(models, list)
    target_id = models[0]["id"]

    resp = await agent.extMethod(
        "model/set",
        {"sessionId": session.sessionId, "modelId": target_id},
    )
    assert resp.get("current") == target_id
    cfg = model_registry.load_models_config()
    assert cfg.get("current") == target_id


@pytest.mark.asyncio
async def test_session_load_replays_history(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    conn = AsyncMock(spec=AgentSideConnection)
    conn.sessionUpdate = AsyncMock()
    agent = ACPAgent(conn)
    session = await agent.newSession(NewSessionRequest(cwd=str(tmp_path), mcpServers=[]))

    note_user = _make_user_chunk(session.sessionId, "hello")
    note_agent = _make_agent_chunk(session.sessionId, "world")
    agent._record_update(note_user)  # type: ignore[attr-defined]
    agent._record_update(note_agent)  # type: ignore[attr-defined]

    conn.sessionUpdate.reset_mock()
    await agent.loadSession(
        LoadSessionRequest(sessionId=session.sessionId, cwd=str(tmp_path), mcpServers=[])
    )

    assert conn.sessionUpdate.await_count == 2
    updates = [call.args[0].update for call in conn.sessionUpdate.await_args_list]  # type: ignore[attr-defined]
    assert any(isinstance(u, AgentMessageChunk) for u in updates)
