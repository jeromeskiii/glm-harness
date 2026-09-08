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


def test_cli_exit_code_three_for_provider_error(monkeypatch) -> None:
    """Provider failures inside the run loop map to exit code 3."""
    from glmharness.errors import ProviderError

    class _BoomLLM:
        async def stream(self, messages, tools=None):
            raise ProviderError("provider went away", retryable=False)
            yield ""  # pragma: no cover


    # Patch MockLLM with a boom-providing LLM by overwriting inside cli.run.
    # We do this by patching the AgentLoop.run to raise from its stream call.
    from glmharness.loop import AgentLoop

    async def boom_run(self, prompt):
        # Trigger ProviderError from the streaming path so cli.run sees it
        # propagate out and the exit-code branch fires.
        self.sessions.append("turn/start", {})
        self.sessions.append("user/message", {"content": prompt})
        raise ProviderError("provider went away", retryable=False)

    monkeypatch.setattr(AgentLoop, "run", boom_run)
    code = main(["--mock", "x", "y"])
    assert code == 3


def test_doctor_mock_exits_zero(capsys) -> None:
    code = main(["--doctor", "--mock", "hi"])
    captured = capsys.readouterr()
    assert code == 0
    assert "ok" in captured.out
    assert "mock" in captured.out
    assert "WARN  request_timeout" not in captured.out


def test_doctor_warns_when_request_timeout_disabled(capsys) -> None:
    code = main(["--doctor", "--mock", "hi", "--request-timeout-s", "0"])
    captured = capsys.readouterr()
    assert code == 0
    assert "WARN  request_timeout" in captured.out


def test_doctor_fails_when_cwd_is_not_a_snapshot(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    code = main(["--doctor"])
    captured = capsys.readouterr()
    assert code == 2
    assert "FAIL" in captured.out


def test_cli_mock_from_env(monkeypatch, capsys) -> None:
    monkeypatch.setenv("GLMH_MOCK", "from-env")
    code = main(["say hi"])
    assert code == 0
    assert capsys.readouterr().out.strip() == "from-env"


def test_cli_requires_prompt_when_stdin_is_not_a_tty(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    code = main(["--mock", "answer"])
    assert code == 2
    assert "prompt is required" in capsys.readouterr().err


def test_parser_accepts_doctor_and_allowlist() -> None:
    args = build_parser().parse_args(["--doctor", "--tool-allowlist", "echo,search"])
    assert args.doctor is True
    assert args.tool_allowlist == "echo,search"


def test_parser_accepts_workspace_and_sandbox(tmp_path) -> None:
    args = build_parser().parse_args(["--workspace", str(tmp_path), "--sandbox", "deny"])
    assert args.workspace == tmp_path
    assert args.sandbox == "deny"


def test_cli_doctor_outputs_workspace_and_sandbox(tmp_path, capsys) -> None:
    code = main(["--doctor", "--mock", "hi", "--workspace", str(tmp_path), "--sandbox", "deny"])
    captured = capsys.readouterr()
    assert code == 0
    assert "workspace:" in captured.out
    assert "sandbox: deny" in captured.out


def test_parser_accepts_api_base_and_key() -> None:
    args = build_parser().parse_args(
        ["--api-base", "http://localhost:8000/v1", "--api-key", "secret", "--model", "custom-glm"]
    )
    assert args.api_base == "http://localhost:8000/v1"
    assert args.api_key == "secret"
    assert args.model == "custom-glm"


def test_cli_doctor_outputs_openai_compatible(capsys) -> None:
    code = main(["--doctor", "--api-base", "http://localhost:8000/v1", "--model", "my-glm"])
    captured = capsys.readouterr()
    assert code == 0
    assert "provider: openai-compatible (http://localhost:8000/v1, model=my-glm)" in captured.out


def test_cli_exit_code_four_for_other_runtime_error(monkeypatch) -> None:
    """Unclassified runtime failures inside the run loop map to exit code 4."""
    from glmharness.loop import AgentLoop

    async def boom_run(self, prompt):
        self.sessions.append("turn/start", {})
        raise RuntimeError("unexpected")

    monkeypatch.setattr(AgentLoop, "run", boom_run)
    code = main(["--mock", "x", "y"])
    assert code == 4
