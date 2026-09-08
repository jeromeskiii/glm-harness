"""Unit tests for interactive terminal REPL."""

from __future__ import annotations

from pathlib import Path

from glmharness import HarnessConfig
from glmharness.repl import run_repl


async def test_repl_commands_and_turn(tmp_path: Path) -> None:
    inputs = [
        "/help",
        "/session",
        "/tools",
        "/clear",
        "/unknown",
        "hello agent",
        "/exit",
    ]
    output_chunks: list[str] = []

    def mock_input(prompt: str) -> str:
        if not inputs:
            return "/exit"
        return inputs.pop(0)

    def mock_output(text: str) -> None:
        output_chunks.append(text)

    config = HarnessConfig(mock="Hello from mock agent!", workspace_dir=tmp_path)
    code = await run_repl(config, input_func=mock_input, output_func=mock_output)

    assert code == 0
    full_output = "".join(output_chunks)
    assert "GLM-5.3-Flash Agent REPL" in full_output
    assert "Available commands:" in full_output
    assert "Registered tools" in full_output
    assert "Conversation history cleared." in full_output
    assert "Unknown command: /unknown" in full_output
    assert "Hello from mock agent!" in full_output
    assert "Goodbye!" in full_output


async def test_repl_eof_handling(tmp_path: Path) -> None:
    output_chunks: list[str] = []

    def mock_input(prompt: str) -> str:
        raise EOFError

    config = HarnessConfig(mock="hi", workspace_dir=tmp_path)
    code = await run_repl(config, input_func=mock_input, output_func=output_chunks.append)
    assert code == 0
    assert any("Goodbye!" in chunk for chunk in output_chunks)
