"""Configuration precedence independent of the developer's shared environment."""

from pathlib import Path

import pytest

from isaac.cli import parse_settings


def test_environment_and_cli_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    config = tmp_path / "isaac"
    config.mkdir()
    (config / ".env").write_text("ISAAC_MODEL=openai:from-file\n", encoding="utf-8")
    monkeypatch.setenv("ISAAC_MODEL", "openai:from-process")
    assert parse_settings([]).models == ("openai:from-process",)
    assert parse_settings(["--model", "openai:from-cli", "--model", "google-gla:other"]).models == (
        "openai:from-cli",
        "google-gla:other",
    )


def test_missing_model_fails_before_starting_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("ISAAC_MODEL", raising=False)
    with pytest.raises(SystemExit) as raised:
        parse_settings([])
    assert raised.value.code == 2
