from pathlib import Path


async def list_files(
    directory: str = ".",
    recursive: bool = True,
) -> dict:
    """List files and directories in a given path.

    Args:
        directory (str): The directory to list. Defaults to current directory.
        recursive (bool): Whether to list subdirectories recursively. Defaults to True.

    Returns:
        dict: A dictionary containing 'content' (string listing) and 'error' (string or None).
    """
    path = Path(directory)
    if not path.exists():
        return {"content": None, "error": f"Directory '{directory}' does not exist."}

    if not path.is_dir():
        return {"content": None, "error": f"'{directory}' is not a directory."}

    try:
        if recursive:
            items = []
            for item in path.rglob("*"):
                item_type = "dir" if item.is_dir() else "file"
                items.append(f"{item.relative_to(path)} [{item_type}]")
            result = "\n".join(sorted(items))
        else:
            items = [
                f"{item.name} [{'dir' if item.is_dir() else 'file'}]" for item in path.iterdir()
            ]
            result = "\n".join(sorted(items))

        return {"content": result, "error": None}
    except Exception as e:
        return {"content": None, "error": f"Error listing directory: {str(e)}"}
