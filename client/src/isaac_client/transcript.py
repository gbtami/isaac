"""Record ACP conversation text independently of terminal rendering."""

import io
import json
import os
import re
from pathlib import Path

from acp import schema

from isaac_client.display import edit_diff, visible

_ANSI = re.compile(
    r"(?:\x1b\[|\x9b)[0-?]*[ -/]*[@-~]"  # CSI
    r"|(?:\x1b\]|\x9d)[^\x07\x1b\x9c]*(?:\x07|\x1b\\|\x9c)"  # OSC
    r"|(?:\x1b[PX^_]|[\x90\x98\x9e\x9f])[^\x1b\x9c]*(?:\x1b\\|\x9c)"  # Control strings
    r"|\x1b[ -/]*[0-~]"
)


class Transcript:
    def __init__(self) -> None:
        self._text = io.StringIO()
        self._speaker: str | None = None

    def message(self, speaker: str, text: str) -> None:
        if speaker != self._speaker:
            self.break_message()
            self._text.write(f"{speaker}: ")
            self._speaker = speaker
        self._text.write(text)

    def break_message(self) -> None:
        if self._speaker is not None:
            self._text.write("\n\n")
            self._speaker = None

    def tool(self, title: str, update: schema.ToolCallStart | schema.ToolCallUpdate) -> None:
        self.break_message()
        status = f" ({update.status.replace('_', ' ')})" if update.status else ""
        self._text.write(f"Tool {update.tool_call_id}: {title}{status}\n")
        for label, data in (("Input", update.raw_input), ("Output", update.raw_output)):
            if data is not None:
                text = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False, indent=2)
                self._text.write(f"{label}:\n{text}\n")
        for block in update.content or []:
            if isinstance(block, schema.FileEditToolCallContent):
                self._text.write(edit_diff(block))
            elif isinstance(block, schema.ContentToolCallContent) and isinstance(
                block.content, schema.TextContentBlock
            ):
                self._text.write(f"{block.content.text}\n")
            elif isinstance(block, schema.TerminalToolCallContent):
                self._text.write(f"Terminal: {block.terminal_id}\n")
        self._text.write("\n")

    def export(self, path: Path) -> None:
        text = visible(_ANSI.sub("", self._text.getvalue()).replace("\r\n", "\n"))
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", errors="replace") as file:
            file.write(text)
            if text and not text.endswith("\n"):
                file.write("\n")
