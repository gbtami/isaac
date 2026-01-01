"""Tests for tool argument coercion helpers."""

from isaac.agent.brain.tool_args import coerce_tool_args


def test_coerce_tool_args_parses_json_object() -> None:
    payload = '{"path": "tests/test_file.py", "content": "hi"}'
    assert coerce_tool_args(payload) == {"path": "tests/test_file.py", "content": "hi"}


def test_coerce_tool_args_falls_back_to_command() -> None:
    payload = "ls -la"
    assert coerce_tool_args(payload) == {"command": "ls -la"}


def test_coerce_tool_args_non_object_json_is_command() -> None:
    payload = '["a", "b"]'
    assert coerce_tool_args(payload) == {"command": payload}


def test_coerce_tool_args_dict_passthrough() -> None:
    payload = {"path": "README.md"}
    assert coerce_tool_args(payload) == {"path": "README.md"}
