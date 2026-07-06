from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

from isaac.agent.ai_types import ToolContext
from isaac.agent.tools.safety import DEFAULT_IGNORES, PathAccessError, is_default_ignored, resolve_workspace_path

import pathspec  # type: ignore

MAX_LIST_ENTRIES = 500
MAX_LIST_CHARS = 20000


async def list_files(
    ctx: ToolContext | None = None,
    directory: str = ".",
    recursive: bool = True,
    cwd: Optional[str] = None,
) -> dict:
    """List files and directories in a given path."""

    _ = ctx
    max_entries = MAX_LIST_ENTRIES
    max_chars = MAX_LIST_CHARS
    try:
        path = resolve_workspace_path(cwd, directory)
    except PathAccessError as exc:
        return {"content": "", "error": str(exc)}

    if not path.exists():
        return {"content": "", "error": f"Directory '{directory}' does not exist."}

    if not path.is_dir():
        return {"content": "", "error": f"'{directory}' is not a directory."}

    try:
        patterns = _load_gitignore_patterns(path)
        matcher = _build_matcher(patterns)
        items = []
        if recursive:
            for root, dirs, files in os.walk(path, followlinks=False):
                rel_root = Path(root).relative_to(path)
                for d in list(dirs):
                    rel_dir = (rel_root / d) if rel_root != Path(".") else Path(d)
                    full_dir = Path(root) / d
                    if _should_skip(rel_dir, is_dir=True) or matcher(rel_dir, is_dir=True):
                        dirs.remove(d)
                        continue
                    # Avoid showing or descending symlinks that point outside the
                    # listed root. os.walk does not follow them by default, but
                    # displaying outside targets still confuses the workspace view.
                    try:
                        full_dir.resolve(strict=False).relative_to(path)
                    except ValueError:
                        dirs.remove(d)
                        continue
                for d in dirs:
                    rel_dir = (rel_root / d) if rel_root != Path(".") else Path(d)
                    if _should_skip(rel_dir, is_dir=True) or matcher(rel_dir, is_dir=True):
                        continue
                    items.append(f"{rel_dir} [dir]")
                for f in files:
                    rel_file = (rel_root / f) if rel_root != Path(".") else Path(f)
                    if _should_skip(rel_file, is_dir=False) or matcher(rel_file, is_dir=False):
                        continue
                    items.append(f"{rel_file} [file]")
        else:
            for item in path.iterdir():
                rel = Path(item.name)
                if _should_skip(rel, is_dir=item.is_dir()) or matcher(rel, is_dir=item.is_dir()):
                    continue
                if item.is_symlink():
                    try:
                        item.resolve(strict=False).relative_to(path)
                    except ValueError:
                        continue
                items.append(f"{rel} [{'dir' if item.is_dir() else 'file'}]")
        items = sorted(items)
        truncated = False
        if len(items) > max_entries:
            items = items[:max_entries]
            truncated = True
        result = "\n".join(items)
        if len(result) > max_chars:
            result = result[:max_chars]
            truncated = True
        if truncated:
            result = f"{result}\n[truncated]"

        response = {"content": result, "error": None}
        if truncated:
            response["truncated"] = True
        return response
    except Exception as e:
        return {"content": "", "error": f"Error listing directory: {str(e)}"}


def _load_gitignore_patterns(root: Path) -> list[str]:
    gitignore_path = root / ".gitignore"
    if not gitignore_path.exists():
        return []
    try:
        with gitignore_path.open(encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    except Exception:
        return []


def _build_matcher(patterns: list[str]) -> Callable[[Path, bool], bool]:
    spec = pathspec.PathSpec.from_lines("gitwildmatch", patterns) if patterns else None

    def _match(path: Path, is_dir: bool) -> bool:
        if spec is None:
            return False
        rel_str = str(path)
        return spec.match_file(rel_str + ("/" if is_dir else ""))

    return _match


def _should_skip(path: Path, is_dir: bool) -> bool:
    """Avoid descending into common cache/virtualenv folders even without .gitignore."""

    _ = is_dir
    return is_default_ignored(path)


__all__ = ["DEFAULT_IGNORES", "MAX_LIST_CHARS", "MAX_LIST_ENTRIES", "list_files"]
