from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from acp import PROTOCOL_VERSION, text_block
from acp.agent.connection import AgentSideConnection
from acp.helpers import session_notification, update_agent_message
from acp.schema import AgentMessageChunk, SessionNotification, UserMessageChunk
from pydantic_ai import messages as ai_messages  # type: ignore
from pydantic_ai.run import AgentRunResultEvent  # type: ignore

from isaac.agent.agent import ACPAgent
from isaac.agent import models as model_registry
from isaac.agent.acp.history import build_chat_history


def _make_user_chunk(session_id: str, text: str) -> SessionNotification:
    return SessionNotification(
        session_id=session_id,
        update=UserMessageChunk(session_update="user_message_chunk", content=text_block(text)),
    )


def _make_agent_chunk(session_id: str, text: str) -> SessionNotification:
    return session_notification(
        session_id,
        update_agent_message(text_block(text)),
    )


@pytest.mark.asyncio
async def test_initialize_advertises_ext_methods(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("HOME", str(tmp_path))
    conn = AsyncMock(spec=AgentSideConnection)
    agent = ACPAgent(conn)

    resp = await agent.initialize(protocol_version=PROTOCOL_VERSION)

    meta = getattr(resp.agent_capabilities, "field_meta", {}) or {}
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
    fn_model_id = model_registry.FUNCTION_MODEL_ID
    minimal_config = {
        "current": fn_model_id,
        "models": {
            fn_model_id: {
                "provider": "function",
                "model": "function",
                "description": "In-process function model for deterministic testing",
            },
            "function:user-function": {
                "provider": "function",
                "model": "user-function",
                "description": "User-visible function model for tests",
            },
        },
    }
    monkeypatch.setattr(model_registry, "DEFAULT_CONFIG", minimal_config)
    model_registry.MODELS_FILE.parent.mkdir(parents=True, exist_ok=True)
    model_registry.MODELS_FILE.write_text(json.dumps(minimal_config), encoding="utf-8")
    conn = AsyncMock(spec=AgentSideConnection)
    agent = ACPAgent(conn)
    session = await agent.new_session(cwd=str(tmp_path), mcp_servers=[])

    listing = await agent.ext_method("model/list", {"session_id": session.session_id})
    assert listing.get("current")
    models = listing.get("models", [])
    assert isinstance(models, list)
    target_id = models[0]["id"]

    resp = await agent.ext_method(
        "model/set",
        {"session_id": session.session_id, "model_id": target_id},
    )
    assert resp.get("current") == target_id
    assert model_registry.current_model_id() == target_id


@pytest.mark.asyncio
async def test_session_load_replays_history(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    conn = AsyncMock(spec=AgentSideConnection)
    conn.session_update = AsyncMock()
    agent = ACPAgent(conn)
    session = await agent.new_session(cwd=str(tmp_path), mcp_servers=[])

    note_user = _make_user_chunk(session.session_id, "hello")
    note_agent = _make_agent_chunk(session.session_id, "world")
    agent._record_update(note_user)  # type: ignore[attr-defined]
    agent._record_update(note_agent)  # type: ignore[attr-defined]

    conn.session_update.reset_mock()
    await agent.load_session(cwd=str(tmp_path), mcp_servers=[], session_id=session.session_id)

    # We may emit an extra usage hint; ensure history is replayed.
    assert conn.session_update.await_count >= 2
    updates = [call.kwargs["update"] for call in conn.session_update.await_args_list]  # type: ignore[attr-defined]
    texts = [getattr(u.content, "text", "") for u in updates if isinstance(u, AgentMessageChunk)]
    assert "world" in " ".join(texts)


@pytest.mark.asyncio
async def test_store_user_prompt_coerces_text_blocks(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    conn = AsyncMock(spec=AgentSideConnection)
    conn.session_update = AsyncMock()
    agent = ACPAgent(conn)
    session = await agent.new_session(cwd=str(tmp_path), mcp_servers=[])

    agent._store_user_prompt(session.session_id, [{"text": "hello world"}])

    chat_history = build_chat_history(agent._session_history.get(session.session_id, []))
    assert any("hello world" in item["content"] for item in chat_history)


class _RecordingRunner:
    def __init__(self, output: str):
        self.output = output
        self.stream_messages: list[dict[str, str]] | None = None
        self._new_messages: list[dict[str, str]] = []

    async def run_stream_events(
        self,
        prompt_text: str,
        messages: list[object] | None = None,
        message_history: list[object] | None = None,
        **_: object,
    ):
        prior = (message_history if message_history is not None else messages) or []
        self.stream_messages = _flatten_history(list(prior))
        history = list(self.stream_messages)
        history.append({"role": "user", "content": prompt_text})
        history.append({"role": "assistant", "content": self.output})
        self._new_messages = history

        async def _gen():
            class _Result:
                def __init__(self, output: str, msgs: list[dict[str, str]]):
                    self.output = output
                    self._msgs = msgs
                    self.usage = None

                def new_messages(self) -> list[dict[str, str]]:
                    return list(self._msgs)

            yield AgentRunResultEvent(result=_Result(self.output, self._new_messages))

        return _gen()

    def new_messages(self) -> list[dict[str, str]]:
        return list(self._new_messages)


def _flatten_history(messages: list[object]) -> list[dict[str, str]]:
    flattened: list[dict[str, str]] = []
    for message in messages:
        if isinstance(message, dict):
            flattened.append(message)
            continue
        if isinstance(message, ai_messages.ModelRequest):
            for part in message.parts:
                if isinstance(part, ai_messages.SystemPromptPart):
                    flattened.append({"role": "system", "content": str(part.content)})
                elif isinstance(part, ai_messages.UserPromptPart):
                    flattened.append({"role": "user", "content": str(part.content)})
        elif isinstance(message, ai_messages.ModelResponse):
            for part in message.parts:
                if isinstance(part, ai_messages.TextPart):
                    flattened.append({"role": "assistant", "content": str(part.content)})
    return flattened


@pytest.mark.asyncio
async def test_history_preserved_across_prompts(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    conn = AsyncMock(spec=AgentSideConnection)
    conn.session_update = AsyncMock()
    executor = _RecordingRunner("done")
    from isaac.agent.brain import session_ops

    def _build(_model_id: str, _register: object, toolsets=None, **kwargs: object) -> object:
        _ = toolsets
        return executor

    monkeypatch.setattr(session_ops, "create_subagent_for_model", _build)
    agent = ACPAgent(conn)
    session = await agent.new_session(cwd=str(tmp_path), mcp_servers=[])

    await agent.prompt(prompt=[text_block("remember this line")], session_id=session.session_id)
    await agent.prompt(prompt=[text_block("what did I say?")], session_id=session.session_id)

    messages = executor.stream_messages or []
    assert any("remember this line" in m.get("content", "") for m in messages)
