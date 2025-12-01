from __future__ import annotations

from pathlib import Path


async def edit_file(file_path: str, new_content: str, create: bool = True, **_: object) -> dict:
    """Overwrite a file with new content.

    Args:
        file_path: Path to the file to edit.
        new_content: Complete replacement content for the file.
        create: Whether to create the file if it does not exist.
    """
    path = Path(file_path)

    if not path.exists() and not create:
        return {"content": None, "error": f"File '{file_path}' does not exist", "returncode": -1}

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(new_content, encoding="utf-8")
        return {
            "content": f"Wrote {len(new_content)} bytes to {file_path}",
            "error": None,
            "returncode": 0,
        }
    except Exception as exc:  # pragma: no cover - unexpected filesystem errors
        return {"content": None, "error": str(exc), "returncode": -1}
