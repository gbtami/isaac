"""Apply a unified diff patch to a file."""

from __future__ import annotations

import asyncio
import tempfile
import textwrap
from pathlib import Path, PurePosixPath
from typing import Optional

from isaac.agent.ai_types import ToolContext
from isaac.agent.tools.safety import (
    BinaryFileError,
    PathAccessError,
    ProtectedPathError,
    ensure_no_symlink_in_write_path,
    ensure_text_target,
    ensure_text_write_size,
    resolve_workspace_path,
    sha256_file,
)


def _strip_patch_target(raw_target: str, strip: int) -> str | None:
    target = raw_target.strip().split()[0] if raw_target.strip() else ""
    if not target or target == "/dev/null":
        return None
    target = target.replace("\\", "/")
    if target.startswith("/"):
        raise ProtectedPathError(f"Patch target is absolute: {target}")
    parts = [part for part in PurePosixPath(target).parts if part not in {"", "."}]
    if ".." in parts:
        raise ProtectedPathError(f"Patch target escapes directory: {target}")
    if strip > len(parts):
        raise ProtectedPathError(f"Patch strip level -p{strip} removes entire target: {target}")
    stripped = parts[strip:]
    if not stripped:
        raise ProtectedPathError(f"Patch target is empty after -p{strip}: {target}")
    return PurePosixPath(*stripped).as_posix()


def _validate_patch_targets(patch_text: str, *, expected_name: str, strip: int) -> None:
    targets: set[str] = set()
    for line in patch_text.splitlines():
        if line.startswith("--- ") or line.startswith("+++ "):
            target = _strip_patch_target(line[4:], strip)
            if target is not None:
                targets.add(target)
    if not targets:
        raise ProtectedPathError("Patch does not contain unified diff file headers")
    unexpected = sorted(target for target in targets if target != expected_name)
    if unexpected:
        raise ProtectedPathError(
            f"Patch targets {', '.join(unexpected)} but this tool call may only modify {expected_name}"
        )


def _prepare_patch_text(patch_text: str, *, expected_name: str, strip: int) -> str:
    candidates = [patch_text]
    dedented = textwrap.dedent(patch_text)
    if dedented != patch_text:
        candidates.append(dedented)
    last_error: ProtectedPathError | None = None
    for candidate in candidates:
        try:
            _validate_patch_targets(candidate, expected_name=expected_name, strip=strip)
        except ProtectedPathError as exc:
            last_error = exc
            continue
        return candidate
    if last_error is not None:
        raise last_error
    raise ProtectedPathError("Patch does not contain unified diff file headers")


async def apply_patch(
    ctx: ToolContext | None = None,
    path: str = "",
    patch: str = "",
    strip: Optional[int] = None,
    cwd: Optional[str] = None,
    expected_sha256: Optional[str] = None,
    session_cwd: str | Path | None = None,
    additional_directories: tuple[str | Path, ...] = (),
    **_: object,
) -> dict:
    """Apply a unified diff patch to a text file using the `patch` command."""

    _ = ctx
    try:
        base = session_cwd or cwd
        ensure_no_symlink_in_write_path(base, path, additional_directories=additional_directories)
        resolved = resolve_workspace_path(base, path, additional_directories=additional_directories)
        ensure_text_target(resolved, base)
    except (PathAccessError, ProtectedPathError, BinaryFileError) as exc:
        return {"path": path, "content": "", "error": str(exc)}

    if not resolved.exists():
        return {"path": path, "content": "", "error": f"File not found: {path}"}
    try:
        old_text = resolved.read_text(encoding="utf-8")
    except Exception:
        old_text = ""
    old_hash = sha256_file(resolved)
    if expected_sha256 is not None and old_hash != expected_sha256:
        return {
            "path": path,
            "content": "",
            "error": "File changed since it was read; expected_sha256 does not match",
            "sha256": old_hash,
        }

    strip_int = strip if strip is not None else 0
    strip_val = str(strip_int)
    try:
        ensure_text_write_size(patch)
        prepared_patch = _prepare_patch_text(patch, expected_name=resolved.name, strip=strip_int)
    except ProtectedPathError as exc:
        return {"path": path, "content": "", "error": str(exc)}

    async def _run_patch(patch_text: str) -> tuple[int, str, str]:
        try:
            tmp_path = ""
            with tempfile.NamedTemporaryFile("w+", delete=False, encoding="utf-8") as tmp:
                tmp.write(patch_text)
                tmp.flush()
                tmp_path = tmp.name

            proc = await asyncio.create_subprocess_exec(
                "patch",
                f"-p{strip_val}",
                "-i",
                tmp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(resolved.parent),
            )
            stdout, stderr = await proc.communicate()
        except FileNotFoundError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            return 1, "", str(exc)
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

        stdout_text = stdout.decode(errors="ignore") if stdout else ""
        stderr_text = stderr.decode(errors="ignore") if stderr else ""
        return proc.returncode or 0, stdout_text, stderr_text

    try:
        rc, stdout_text, stderr_text = await _run_patch(prepared_patch)
        if rc != 0:
            error_msg = stderr_text or stdout_text or f"Patch failed ({rc})"
            return {"path": path, "content": stdout_text, "error": error_msg}
        try:
            new_text = resolved.read_text(encoding="utf-8")
        except Exception:
            new_text = ""
        return {
            "path": path,
            "content": stdout_text or "Patch applied",
            "error": None,
            "new_text": new_text,
            "old_text": old_text,
            "old_sha256": old_hash,
            "sha256": sha256_file(resolved),
        }
    except FileNotFoundError:
        return {"path": path, "content": "", "error": "`patch` command not found"}
