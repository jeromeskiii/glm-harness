"""Tests for the CLI's argv parsing, config layering, and exit codes."""

from __future__ import annotations

import io
import sys

import pytest

from glmharness.cli import build_parser, main


def test_parser_accepts_prompt_and_mock() -> None:
    args = build_parser().parse_args(["--mock", "hi", "say hi"])
    assert args.mock == "hi"
    assert args.prompt == "say hi"


def test_parser_rejects_bad_reasoning_effort() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--reasoning-effort", "medium", "x"])


def test_parser_rejects_corrupt_policy(capsys) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--corrupt-policy", "ignore", "x"])


def test_version_flag_prints() -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--version"])
    assert exc.value.code == 0


def test_cli_exit_code_zero_for_mock(monkeypatch, capsys) -> None:
    # avoid prompting
    monkeypatch.setattr(sys, "stdin", io.StringIO("ignored\n"))
    code = main(["--mock", "answer", "x"])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.out.strip() == "answer"


def test_cli_exit_code_two_on_bad_config(monkeypatch) -> None:
    monkeypatch.setenv("GLMH_REASONING_EFFORT", "medium")
    code = main(["--mock", "x", "y"])
    assert code == 2


def test_cli_exit_code_130_on_keyboard_interrupt(monkeypatch) -> None:
    from glmharness import cli as cli_mod

    async def raise_kbi(_):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli_mod, "run", raise_kbi)
    code = main(["--mock", "x", "y"])
    assert code == 130
