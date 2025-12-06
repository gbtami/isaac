import os
from pathlib import Path
from typing import Callable, Optional

import pathspec  # type: ignore


def _resolve(base: Optional[str], target: str) -> Path:
    p = Path(target)
    if p.is_absolute():
        return p
    return Path(base or Path.cwd()) / p


async def list_files(
    directory: str = ".",
    recursive: bool = True,
    cwd: Optional[str] = None,
) -> dict:
    """List files and directories in a given path.

    Args:
        directory (str): The directory to list. Defaults to current directory.
        recursive (bool): Whether to list subdirectories recursively. Defaults to True.

    Returns:
        dict: A dictionary containing 'content' (string listing) and 'error' (string or None).
    """
    max_entries = 500
    max_chars = 20000
    path = _resolve(cwd, directory)
    if not path.exists():
        return {"content": None, "error": f"Directory '{directory}' does not exist."}

    if not path.is_dir():
        return {"content": None, "error": f"'{directory}' is not a directory."}

    try:
        patterns = _load_gitignore_patterns(path)
        matcher = _build_matcher(patterns)
        items = []
        if recursive:
            for root, dirs, files in os.walk(path):
                rel_root = Path(root).relative_to(path)
                # Prune ignored dirs to avoid descending into them.
                for d in list(dirs):
                    rel_dir = (rel_root / d) if rel_root != Path(".") else Path(d)
                    if _should_skip(rel_dir, is_dir=True) or matcher(rel_dir, is_dir=True):
                        dirs.remove(d)
                # Add remaining dirs and files
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
        return {"content": None, "error": f"Error listing directory: {str(e)}"}


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
        # Ensure directories match when patterns include trailing slashes semantics.
        return spec.match_file(rel_str + ("/" if is_dir else ""))

    return _match


_DEFAULT_IGNORES = {
    ".git",
    ".hg",
    ".svn",
    ".cache",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".tox",
    ".nox",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".uv-cache",
    "node_modules",
}


def _should_skip(path: Path, is_dir: bool) -> bool:
    """Avoid descending into common cache/virtualenv folders even without .gitignore."""

    parts = set(path.parts)
    return any(part in _DEFAULT_IGNORES for part in parts)
