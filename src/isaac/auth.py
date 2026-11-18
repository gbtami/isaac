"""Small CLI and storage bridge around Pydantic AI's Codex OAuth provider."""

import asyncio
import os
import sys
import tempfile
import webbrowser
from collections.abc import Callable
from pathlib import Path

from pydantic import TypeAdapter, ValidationError
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import Model
from pydantic_ai.models.openai_codex import OpenAICodexModel
from pydantic_ai.providers.openai_codex import (
    OpenAICodexCredentials,
    OpenAICodexCredentialSource,
    OpenAICodexOAuthFlow,
    OpenAICodexProvider,
)

from isaac.config import env_file

_CREDENTIALS = TypeAdapter(OpenAICodexCredentials)


class CodexCredentialStore(OpenAICodexCredentialSource):
    """Persist login and rotated tokens without touching the Codex CLI's files."""

    def __init__(self) -> None:
        self.path = env_file().parent / "auth" / "openai-codex.json"

    async def load(self) -> OpenAICodexCredentials:
        return await asyncio.to_thread(self._load)

    def _load(self) -> OpenAICodexCredentials:
        try:
            data = self.path.read_bytes()
        except FileNotFoundError:
            raise UserError("No Isaac Codex login found. Run `isaac login`, then restart Isaac.") from None
        try:
            credentials = _CREDENTIALS.validate_json(data, strict=True)
            if not all((credentials.access_token, credentials.refresh_token, credentials.account_id)):
                raise ValueError("Empty credential field")
        except (ValidationError, ValueError):
            # Validation errors can contain token values; never include their details in diagnostics.
            raise UserError(f"Invalid Codex credentials at `{self.path}`. Run `isaac login` again.") from None
        return credentials

    async def save(self, credentials: OpenAICodexCredentials) -> None:
        await asyncio.to_thread(self._save, credentials)

    def _save(self, credentials: OpenAICodexCredentials) -> None:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.path.parent.chmod(0o700)
        # A private temporary file and atomic replacement keep readers from seeing partial tokens.
        descriptor, filename = tempfile.mkstemp(dir=self.path.parent, prefix=".codex-")
        temporary = Path(filename)
        try:
            with os.fdopen(descriptor, "wb") as file:
                file.write(_CREDENTIALS.dump_json(credentials))
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)


async def login(*, open_browser: bool = True, timeout: float = 300) -> None:
    """Run the upstream browser/callback flow independently of any ACP client."""
    flow = OpenAICodexOAuthFlow()
    url = flow.authorization_url()
    print("Sign in to ChatGPT/Codex. Waiting up to five minutes.", file=sys.stderr)
    print(f"Open this URL in your browser:\n{url}", file=sys.stderr)
    print("The browser must reach localhost:1455 on this machine.", file=sys.stderr)

    async def launch_browser() -> None:
        try:
            if not await asyncio.to_thread(webbrowser.open, url):
                print("No browser opened; use the URL above.", file=sys.stderr)
        except (webbrowser.Error, OSError):
            print("Could not open a browser; use the URL above.", file=sys.stderr)

    browser = asyncio.create_task(launch_browser()) if open_browser else None
    try:
        try:
            async with asyncio.timeout(timeout):
                credentials = await flow.exchange_code_from_callback()
        except TimeoutError:
            raise UserError("Codex login timed out. Run `isaac login` to try again.") from None
        await CodexCredentialStore().save(credentials)
    finally:
        if browser is not None:
            browser.cancel()
            await asyncio.gather(browser, return_exceptions=True)
    print("Codex login saved. Select an openai-codex: model and start or restart Isaac.", file=sys.stderr)


def model_resolver() -> Callable[[str], Model | str]:
    """Share one Codex provider per server; let core resolve all other model IDs."""
    store = CodexCredentialStore()
    provider: OpenAICodexProvider | None = None

    def resolve(model_id: str) -> Model | str:
        nonlocal provider
        prefix, _, name = model_id.partition(":")
        if prefix != "openai-codex":
            return model_id
        if not name:
            raise UserError("Use an openai-codex: model identifier with a model name.")
        if provider is None:
            if store.path.exists():
                provider = OpenAICodexProvider(credential_source=store)
            else:
                # Core supports an existing Codex CLI login as a read-only fallback.
                try:
                    provider = OpenAICodexProvider()
                except UserError:
                    raise UserError("No usable Codex login found. Run `isaac login`, then restart Isaac.") from None
        return OpenAICodexModel(name, provider=provider)

    return resolve
