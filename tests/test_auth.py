"""Test the upstream OAuth bridge with fake credentials and no browser or provider network."""

import asyncio
import json
import os
import stat
import webbrowser
from pathlib import Path
from urllib.parse import parse_qs

import httpx2
import pytest
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.openai_codex import OpenAICodexModel
from pydantic_ai.providers.openai_codex import OpenAICodexCredentials, OpenAICodexOAuthFlow, OpenAICodexProvider

from isaac import auth
from isaac.cli import main


@pytest.fixture(autouse=True)
def isolate_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))


def credentials(token: str = "fake-access") -> OpenAICodexCredentials:
    return OpenAICodexCredentials(access_token=token, refresh_token=f"refresh-{token}", account_id="fake-account")


async def test_credentials_are_private_atomic_and_reopenable(monkeypatch: pytest.MonkeyPatch) -> None:
    store = auth.CodexCredentialStore()
    original = credentials()
    await store.save(original)
    assert await auth.CodexCredentialStore().load() == original
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600
    assert stat.S_IMODE(store.path.parent.stat().st_mode) == 0o700

    def fail_replace(_source: Path, _destination: Path) -> None:
        raise OSError("Fake disk failure")

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", fail_replace)
        with pytest.raises(OSError, match="Fake disk failure"):
            await store.save(credentials("replacement"))
    assert await store.load() == original
    assert list(store.path.parent.iterdir()) == [store.path]
    await store.save(credentials("replacement"))
    assert await auth.CodexCredentialStore().load() == credentials("replacement")


async def test_invalid_credentials_are_reported_without_token_values() -> None:
    store = auth.CodexCredentialStore()
    with pytest.raises(UserError, match="isaac login"):
        await store.load()
    await store.save(credentials())
    malformed = b'{"access_token":"secret-token-value","refresh_token":42,"account_id":"fake"}'
    store.path.write_bytes(malformed)
    with pytest.raises(UserError, match="Invalid Codex credentials") as raised:
        await store.load()
    assert "secret-token-value" not in str(raised.value)
    assert store.path.read_bytes() == malformed


async def test_login_uses_core_flow_and_keeps_stdout_clean(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    opened = asyncio.Event()
    loop = asyncio.get_running_loop()

    def browser(url: str) -> bool:
        assert url.startswith("https://auth.openai.com/oauth/authorize?")
        loop.call_soon_threadsafe(opened.set)
        return False  # Manual opening remains possible when no browser helper is installed.

    async def callback(_flow: OpenAICodexOAuthFlow) -> OpenAICodexCredentials:
        await opened.wait()
        return credentials()

    monkeypatch.setattr(webbrowser, "open", browser)
    monkeypatch.setattr(OpenAICodexOAuthFlow, "exchange_code_from_callback", callback)
    await auth.login(timeout=2)
    assert await auth.CodexCredentialStore().load() == credentials()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Codex login saved" in captured.err
    assert "fake-access" not in captured.err


async def test_login_timeout_preserves_previous_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    await auth.CodexCredentialStore().save(credentials())
    cancelled = asyncio.Event()

    async def callback(_flow: OpenAICodexOAuthFlow) -> OpenAICodexCredentials:
        try:
            await asyncio.Future[None]()
        finally:
            cancelled.set()
        raise AssertionError("Callback should be cancelled")

    monkeypatch.setattr(OpenAICodexOAuthFlow, "exchange_code_from_callback", callback)
    awaitable = auth.login(open_browser=False, timeout=0.01)
    with pytest.raises(UserError, match="timed out"):
        await awaitable
    assert cancelled.is_set()
    assert await auth.CodexCredentialStore().load() == credentials()


def test_login_cli_needs_no_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ISAAC_MODEL", raising=False)
    calls: list[bool] = []

    async def login(*, open_browser: bool) -> None:
        calls.append(open_browser)

    monkeypatch.setattr(auth, "login", login)
    main(["login", "--no-browser"])
    assert calls == [False]


@pytest.mark.parametrize("failure", [UserError("Sign-in denied"), OSError("Port already in use")])
def test_login_cli_errors_are_actionable(
    failure: Exception, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def login(*, open_browser: bool) -> None:
        raise failure

    monkeypatch.setattr(auth, "login", login)
    with pytest.raises(SystemExit) as raised:
        main(["login"])
    assert raised.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert str(failure) in captured.err
    assert "Traceback" not in captured.err


async def test_model_resolver_shares_provider_and_defers_credentials() -> None:
    resolve = auth.model_resolver()
    assert resolve("anthropic:example") == "anthropic:example"
    with pytest.raises(UserError, match="isaac login"):
        resolve("openai-codex:example")
    await auth.CodexCredentialStore().save(credentials())
    first = resolve("openai-codex:first")
    second = resolve("openai-codex:second")
    assert isinstance(first, OpenAICodexModel)
    assert isinstance(second, OpenAICodexModel)
    assert first.model_name == "first"
    assert second.model_name == "second"
    assert first.provider is second.provider
    assert isinstance(first.provider, OpenAICodexProvider)
    async with first.provider:
        pass


async def test_codex_cli_login_is_a_read_only_fallback(tmp_path: Path) -> None:
    cli_file = tmp_path / "codex" / "auth.json"
    cli_file.parent.mkdir()
    data = {"tokens": {"access_token": "cli-token", "refresh_token": "cli-refresh", "account_id": "cli-account"}}
    cli_file.write_text(json.dumps(data), encoding="utf-8")
    original = cli_file.read_bytes()
    model = auth.model_resolver()("openai-codex:example")
    assert isinstance(model, OpenAICodexModel)
    assert isinstance(model.provider, OpenAICodexProvider)
    async with model.provider:
        assert model.provider.credentials.access_token == "cli-token"
    assert cli_file.read_bytes() == original
    assert not auth.CodexCredentialStore().path.exists()


async def test_core_refresh_persists_rotated_credentials() -> None:
    """A real provider's 401 refresh writes through Isaac's public credential source."""
    store = auth.CodexCredentialStore()
    await store.save(credentials())
    requests: list[str] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        requests.append(request.url.path)
        if request.url.path == "/oauth/token":
            grant = parse_qs(request.content.decode())
            assert grant["refresh_token"] == ["refresh-fake-access"]
            return httpx2.Response(200, json={"access_token": "rotated", "refresh_token": "rotated-refresh"})
        assert request.headers["chatgpt-account-id"] == "fake-account"
        if request.headers["Authorization"] == "Bearer fake-access":
            return httpx2.Response(401, json={"error": "expired"})
        assert request.headers["Authorization"] == "Bearer rotated"
        return httpx2.Response(200, json={"ok": True})

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as client:
        async with OpenAICodexProvider(credential_source=store, http_client=client):
            response = await client.get("https://chatgpt.com/backend-api/codex/usage")
            assert response.status_code == 200
    saved = await auth.CodexCredentialStore().load()
    assert saved.access_token == "rotated"
    assert saved.refresh_token == "rotated-refresh"
    assert requests == ["/backend-api/codex/usage", "/oauth/token", "/backend-api/codex/usage"]
