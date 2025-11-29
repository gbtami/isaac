import asyncio
import asyncio.subprocess as aio_subprocess
import contextlib
import logging
import os
import sys
from pathlib import Path
from typing import Iterable

from prompt_toolkit import PromptSession  # type: ignore
from prompt_toolkit.key_binding import KeyBindings  # type: ignore
from prompt_toolkit.shortcuts import radiolist_dialog  # type: ignore

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
)

from isaac import models as model_registry


class ExampleClient(Client):
    def __init__(self) -> None:
        self._last_prompt = ""

    async def requestPermission(self, params):  # type: ignore[override]
        try:
            selection = await radiolist_dialog(
                title="Permission required",
                text="Select an option",
                values=[(opt.optionId, getattr(opt, "label", opt.optionId)) for opt in params.options],
            ).run_async()
        except Exception:
            selection = params.options[0].optionId if params.options else "default"
        return RequestPermissionResponse(outcome=AllowedOutcome(optionId=selection))

    async def writeTextFile(self, params):  # type: ignore[override]
        raise RequestError.method_not_found("fs/write_text_file")

    async def readTextFile(self, params):  # type: ignore[override]
        raise RequestError.method_not_found("fs/read_text_file")

    async def createTerminal(self, params):  # type: ignore[override]
        raise RequestError.method_not_found("terminal/create")

    async def terminalOutput(self, params):  # type: ignore[override]
        raise RequestError.method_not_found("terminal/output")

    async def releaseTerminal(self, params):  # type: ignore[override]
        raise RequestError.method_not_found("terminal/release")

    async def waitForTerminalExit(self, params):  # type: ignore[override]
        raise RequestError.method_not_found("terminal/wait_for_exit")

    async def killTerminal(self, params):  # type: ignore[override]
        raise RequestError.method_not_found("terminal/kill")

    async def sessionUpdate(self, params: SessionNotification) -> None:
        update = params.update
        if not isinstance(update, AgentMessageChunk):
            return

        content = update.content
        text: str
        if isinstance(content, TextContentBlock):
            text = content.text
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

        print(f"| Agent: {text}")

    async def extMethod(self, method: str, params: dict) -> dict:  # noqa: ARG002
        raise RequestError.method_not_found(method)

    async def extNotification(self, method: str, params: dict) -> None:  # noqa: ARG002
        raise RequestError.method_not_found(method)


async def read_console(prompt: str) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: input(prompt))


async def _select_model_interactive() -> str | None:
    models = model_registry.list_user_models()
    current = model_registry.load_models_config().get("current", "function-model")
    try:
        selection = await radiolist_dialog(
            title="Select model",
            text=f"Current: {current}",
            values=[(mid, f"{mid} ({meta.get('description', '')})") for mid, meta in models.items()],
        ).run_async()
    except Exception:
        return None
    return selection


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
        event.app.exit(result=CANCEL_TOKEN)

    session = PromptSession("> ", key_bindings=kb)
    current_mode = "ask"
    permission_always = False

    while True:
        try:
            line = await session.prompt_async()
            if line == CANCEL_TOKEN:
                await conn.cancel(CancelNotification(sessionId=session_id))
                print("[cancelled]")
                continue
        except EOFError:
            break
        except KeyboardInterrupt:
            print("", file=sys.stderr)
            break

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

        if current_mode == "ask":
            if permission_always:
                pass  # proceed without prompting again
            else:
                try:
                    choice = await radiolist_dialog(
                        title="Permission required",
                        text="Allow this action?",
                        values=[
                            ("once", "Yes, proceed (once)"),
                            ("always", "Yes, don't ask again for this session"),
                            ("deny", "No, revise the request"),
                        ],
                    ).run_async()
                except Exception:
                    choice = "deny"
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
            clientCapabilities=ClientCapabilities(),
            clientInfo=Implementation(name="example-client", title="Example Client", version="0.1.0"),
        )
    )
    session = await conn.newSession(NewSessionRequest(mcpServers=[], cwd=os.getcwd()))

    await interactive_loop(conn, session.sessionId)

    if proc.returncode is None:
        proc.terminate()
        with contextlib.suppress(ProcessLookupError):
            await proc.wait()

    return 0


async def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: python -m isaac.client AGENT_PROGRAM [ARGS...]", file=sys.stderr)
        return 2

    program = argv[1]
    args = argv[2:]
    return await run_client(program, args)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(sys.argv)))
