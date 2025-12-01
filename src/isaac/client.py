"""Interactive ACP client with REPL, tool calls, and terminal support.

ACP overview: https://agentclientprotocol.com/overview/introduction
Terminals: https://agentclientprotocol.com/protocol/terminals
"""

import asyncio
import contextlib
import logging
import os
import sys
from pathlib import Path
from typing import Iterable

import asyncio.subprocess as aio_subprocess

from prompt_toolkit import PromptSession  # type: ignore
from prompt_toolkit.key_binding import KeyBindings  # type: ignore
from prompt_toolkit.patch_stdout import patch_stdout  # type: ignore
from acp import (
    CancelNotification,
    Client,
    ClientSideConnection,
    InitializeRequest,
    NewSessionRequest,
    PROTOCOL_VERSION,
    PromptRequest,
    RequestError,
    RequestPermissionResponse,
    SessionNotification,
    SetSessionModeRequest,
    SetSessionModelRequest,
    text_block,
    CreateTerminalRequest,
    CreateTerminalResponse,
    TerminalOutputRequest,
    TerminalOutputResponse,
    WaitForTerminalExitRequest,
    WaitForTerminalExitResponse,
    KillTerminalCommandRequest,
    KillTerminalCommandResponse,
    ReleaseTerminalRequest,
    ReleaseTerminalResponse,
)
from acp.schema import (
    AgentMessageChunk,
    AllowedOutcome,
    AudioContentBlock,
    ClientCapabilities,
    EmbeddedResourceContentBlock,
    ImageContentBlock,
    Implementation,
    ResourceContentBlock,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
)

from isaac import models as model_registry
from isaac.client_terminal import ClientTerminalManager


class ExampleClient(Client):
    def __init__(self) -> None:
        self._last_prompt = ""
        self._terminal_manager = ClientTerminalManager()

    async def requestPermission(self, params):  # type: ignore[override]
        try:
            for idx, opt in enumerate(params.options, start=1):
                label = getattr(opt, "label", opt.optionId)
                print(f"{idx}) {label}")
            choice = input("Permission choice (number): ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(params.options):
                selection = params.options[int(choice) - 1].optionId
            else:
                selection = params.options[0].optionId if params.options else "default"
        except Exception:
            selection = params.options[0].optionId if params.options else "default"
        return RequestPermissionResponse(outcome=AllowedOutcome(optionId=selection))

    async def writeTextFile(self, params):  # type: ignore[override]
        raise RequestError.method_not_found("fs/write_text_file")

    async def readTextFile(self, params):  # type: ignore[override]
        raise RequestError.method_not_found("fs/read_text_file")

    async def createTerminal(
        self, params: CreateTerminalRequest
    ) -> CreateTerminalResponse:  # type: ignore[override]
        return await self._terminal_manager.create_terminal(params)

    async def terminalOutput(
        self, params: TerminalOutputRequest
    ) -> TerminalOutputResponse:  # type: ignore[override]
        return await self._terminal_manager.terminal_output(params)

    async def releaseTerminal(
        self, params: ReleaseTerminalRequest
    ) -> ReleaseTerminalResponse:  # type: ignore[override]
        return await self._terminal_manager.release_terminal(params)

    async def waitForTerminalExit(
        self, params: WaitForTerminalExitRequest
    ) -> WaitForTerminalExitResponse:  # type: ignore[override]
        return await self._terminal_manager.wait_for_terminal_exit(params)

    async def killTerminalCommand(
        self, params: KillTerminalCommandRequest
    ) -> KillTerminalCommandResponse:  # type: ignore[override]
        return await self._terminal_manager.kill_terminal(params)

    async def sessionUpdate(self, params: SessionNotification) -> None:
        update = params.update
        if isinstance(update, ToolCallStart):
            print(f"| Tool[start]: {getattr(update, 'title', '')}")
            return
        if isinstance(update, ToolCallProgress):
            raw_out = getattr(update, "rawOutput", {}) or {}
            if update.status == "completed":
                rc = raw_out.get("returncode")
                err = raw_out.get("error")
                truncated = raw_out.get("truncated")
                summary_bits = []
                if rc is not None:
                    summary_bits.append(f"rc={rc}")
                if err:
                    summary_bits.append(f"error={err}")
                if truncated:
                    summary_bits.append("truncated")
                summary = " ".join(summary_bits) if summary_bits else "done"
                print(f"| Tool[{update.status}]: {summary}")
            else:
                text = raw_out.get("content") or raw_out.get("error") or ""
                print(f"| Tool[{update.status}]: {text}")
            return
        if not isinstance(update, AgentMessageChunk):
            return

        content = update.content
        text: str
        prefix = ""
        if isinstance(content, TextContentBlock):
            text = content.text
            if prefix:
                print(f"| Agent: {prefix} {text}")
            else:
                print(text, end="", flush=True)
            return
        elif isinstance(content, ImageContentBlock):
            text = "<image>"
        elif isinstance(content, AudioContentBlock):
            text = "<audio>"
        elif isinstance(content, ResourceContentBlock):
            text = content.uri or "<resource>"
        elif isinstance(content, EmbeddedResourceContentBlock):
            text = "<resource>"
        else:
            text = "<content>"

        print(f"| Agent: {prefix} {text}")


async def read_console(prompt: str) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: input(prompt))


async def _select_model_interactive() -> str | None:
    models = model_registry.list_user_models()
    current = model_registry.load_models_config().get("current", "function-model")
    if not models:
        return None

    print(f"Current model: {current}")
    ordered = list(models.items())
    for idx, (mid, meta) in enumerate(ordered, start=1):
        desc = meta.get("description", "")
        print(f"{idx}) {mid}: {desc}")

    loop = asyncio.get_running_loop()
    try:
        choice = await loop.run_in_executor(None, lambda: input("Select model (number): ").strip())
    except Exception:
        return None
    if not choice.isdigit():
        return None
    num = int(choice)
    if 1 <= num <= len(ordered):
        return ordered[num - 1][0]
    return None


async def _run_tests() -> int:
    proc = await asyncio.create_subprocess_shell(
        "uv run pytest",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    assert proc.stdout is not None
    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        print(line.decode().rstrip())
    return await proc.wait()


async def interactive_loop(conn: ClientSideConnection, session_id: str) -> None:
    kb = KeyBindings()
    CANCEL_TOKEN = "__CANCEL__"

    @kb.add("escape")
    def _(event):  # type: ignore
        if not event.app.is_done:
            event.app.exit(result=CANCEL_TOKEN)

    session = PromptSession(key_bindings=kb)
    current_mode = "ask"
    current_model = model_registry.load_models_config().get("current", "function-model")
    permission_always = False

    with patch_stdout():
        while True:
            try:
                line = await session.prompt_async(f"{current_mode}|{current_model}> ")
                if line == CANCEL_TOKEN:
                    await conn.cancel(CancelNotification(sessionId=session_id))
                    print("[cancelled]")
                    continue
            except EOFError:
                break
            except KeyboardInterrupt:
                print("", file=sys.stderr)
                continue

            if not line:
                continue

            # Handle slash commands locally
            if line.startswith("/models"):
                cfg = model_registry.load_models_config()
                current = cfg.get("current", "function-model")
                print(f"Current model: {current}")
                for mid, meta in model_registry.list_user_models().items():
                    desc = meta.get("description", "")
                    print(f"- {mid}: {desc}")
                continue

            if line.startswith("/model"):
                parts = line.split()
                selection = parts[1] if len(parts) > 1 else await _select_model_interactive()
                if not selection:
                    print("[model unchanged]")
                    continue
                try:
                    model_registry.set_current_model(selection)
                except ValueError as exc:
                    print(f"[{exc}]")
                    continue
                current_model = selection
                await conn.setSessionModel(
                    SetSessionModelRequest(sessionId=session_id, modelId=selection)
                )
                print(f"[model set to {selection}]")
                continue

            if line.startswith("/mode"):
                parts = line.split()
                if len(parts) == 1:
                    print("Usage: /mode <ask|yolo>")
                    continue
                await conn.setSessionMode(
                    SetSessionModeRequest(sessionId=session_id, modeId=parts[1])
                )
                current_mode = parts[1]
                # Reset per-session permission policy when switching modes
                permission_always = False
                print(f"[mode set to {current_mode}]")
                continue

            if line.startswith("/test"):
                print("[running tests: uv run pytest]")
                await _run_tests()
                continue

            if line in ("/exit", "/quit", "exit", "quit"):
                print("[exiting]")
                break

            if current_mode == "ask":
                if permission_always:
                    pass  # proceed without prompting again
                else:
                    print("Permission required. Choose:")
                    print("1) Yes, proceed (once)")
                    print("2) Yes, don't ask again for this session")
                    print("3) No, revise the request")
                    loop = asyncio.get_running_loop()
                    try:
                        raw = await loop.run_in_executor(None, lambda: input("Choice [1-3]: ").strip())
                    except Exception:
                        raw = "3"
                    choice = {"1": "once", "2": "always", "3": "deny"}.get(raw, "deny")
                    if choice == "deny":
                        print("[prompt cancelled; please rephrase]")
                        continue
                    if choice == "always":
                        permission_always = True

            try:
                await conn.prompt(
                    PromptRequest(
                        sessionId=session_id,
                        prompt=[text_block(line)],
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logging.error("Prompt failed: %s", exc)


async def run_client(program: str, args: Iterable[str]) -> int:
    logging.basicConfig(level=logging.INFO)

    program_path = Path(program)
    spawn_program = program
    spawn_args = list(args)

    if program_path.exists() and not os.access(program_path, os.X_OK):
        spawn_program = sys.executable
        spawn_args = [str(program_path), *spawn_args]

    proc = await asyncio.create_subprocess_exec(
        spawn_program,
        *spawn_args,
        stdin=aio_subprocess.PIPE,
        stdout=aio_subprocess.PIPE,
    )

    if proc.stdin is None or proc.stdout is None:
        print("Agent process does not expose stdio pipes", file=sys.stderr)
        return 1

    client_impl = ExampleClient()
    conn = ClientSideConnection(lambda _agent: client_impl, proc.stdin, proc.stdout)

    await conn.initialize(
        InitializeRequest(
            protocolVersion=PROTOCOL_VERSION,
            clientCapabilities=ClientCapabilities(terminal=True),
            clientInfo=Implementation(name="example-client", title="Example Client", version="0.1.0"),
        )
    )
    session = await conn.newSession(NewSessionRequest(mcpServers=[], cwd=os.getcwd()))

    try:
        await interactive_loop(conn, session.sessionId)
        return 0
    except KeyboardInterrupt:
        return 130
    finally:
        if proc.returncode is None:
            proc.terminate()
            with contextlib.suppress(ProcessLookupError):
                await proc.wait()


async def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: python -m isaac.client AGENT_PROGRAM [ARGS...]", file=sys.stderr)
        return 2

    program = argv[1]
    args = argv[2:]
    return await run_client(program, args)


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main(sys.argv)))
    except KeyboardInterrupt:
        raise SystemExit(130)
