"""Local OAuth callback server for loopback flows."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable
from urllib.parse import parse_qs, urlparse


class OAuthCallbackError(RuntimeError):
    """Raised when the OAuth callback returns an error or mismatched state."""


@dataclass(frozen=True)
class OAuthCallbackServerConfig:
    path: str
    state: str
    success_html: str
    error_html: Callable[[str], str]
    listen_host: str = "localhost"
    redirect_host: str = "localhost"
    port: int = 0


class OAuthCallbackServer:
    """Run a loopback HTTP server to capture OAuth authorization codes."""

    def __init__(self, config: OAuthCallbackServerConfig) -> None:
        self._config = config
        self._loop = asyncio.get_running_loop()
        self._future: asyncio.Future[str] = self._loop.create_future()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._redirect_uri: str | None = None

    @property
    def redirect_uri(self) -> str:
        if not self._redirect_uri:
            raise RuntimeError("OAuth callback server not started.")
        return self._redirect_uri

    def start(self) -> str:
        if self._server is not None:
            return self.redirect_uri

        config = self._config
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path != config.path:
                    self.send_response(404)
                    self.end_headers()
                    return

                query = parse_qs(parsed.query)
                error = query.get("error", [None])[0]
                error_description = query.get("error_description", [None])[0]
                if error:
                    message = error_description or error
                    outer._set_exception(OAuthCallbackError(message))
                    self._send_html(config.error_html(message), status=400)
                    return

                state = query.get("state", [None])[0]
                if not state or state != config.state:
                    message = "OAuth state mismatch."
                    outer._set_exception(OAuthCallbackError(message))
                    self._send_html(config.error_html(message), status=400)
                    return

                code = query.get("code", [None])[0]
                if not code:
                    message = "Missing authorization code."
                    outer._set_exception(OAuthCallbackError(message))
                    self._send_html(config.error_html(message), status=400)
                    return

                outer._set_result(code)
                self._send_html(config.success_html, status=200)

            def log_message(self, _format: str, *_args: object) -> None:  # noqa: D401,N802
                return

            def _send_html(self, body: str, status: int) -> None:
                self.send_response(status)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(body.encode("utf-8"))

        server = ThreadingHTTPServer((config.listen_host, config.port), Handler)
        self._server = server
        self._redirect_uri = f"http://{config.redirect_host}:{server.server_address[1]}{config.path}"

        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self._thread = thread
        return self.redirect_uri

    async def wait_for_code(self, timeout_s: float) -> str:
        try:
            return await asyncio.wait_for(self._future, timeout=timeout_s)
        finally:
            self.close()

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        self._thread = None

    def _set_result(self, code: str) -> None:
        if self._future.done():
            return
        self._loop.call_soon_threadsafe(self._future.set_result, code)

    def _set_exception(self, exc: Exception) -> None:
        if self._future.done():
            return
        self._loop.call_soon_threadsafe(self._future.set_exception, exc)
