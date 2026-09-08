"""Tests for stop-slop prose analysis, rewriting, skill trigger, and tool integration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from glmharness import (
    BuiltinToolsPlugin,
    Context,
    RiskLevel,
    SkillCatalog,
    SkillsPlugin,
    StopSlopEngine,
    ToolRegistry,
    default_skills,
    make_stop_slop_analyze_tool,
    make_stop_slop_examples_tool,
    make_stop_slop_rewrite_tool,
    make_stop_slop_rules_tool,
)
from glmharness.cli import main
from glmharness.config import HarnessConfig
from glmharness.repl import run_repl


def test_stop_slop_engine_clean_text() -> None:
    engine = StopSlopEngine()
    clean_text = "Building products is hard. Technology is manageable. People aren't."
    report = engine.analyze(clean_text)

    assert report["total_score"] >= 45
    assert report["passes_threshold"] is True
    assert report["violations_count"] == 0
    assert report["dimensions"]["directness"] == 10
    assert report["dimensions"]["trust"] == 10


def test_stop_slop_engine_sloppy_text() -> None:
    engine = StopSlopEngine()
    sloppy = (
        "Here's the thing: in today's fast-paced landscape, we need to navigate challenges "
        "and lean into discomfort. Not because the tech is complex, but because people are complex. "
        "Make no mistake—this is really, literally a game-changer. Let that sink in."
    )
    report = engine.analyze(sloppy)

    assert report["passes_threshold"] is False
    assert report["total_score"] < 35
    assert report["violations_count"] >= 6

    categories = {v["category"] for v in report["violations"]}
    assert "throat_clearing" in categories
    assert "business_jargon" in categories
    assert "emphasis_crutch" in categories
    assert "adverb" in categories
    assert "em_dash" in categories


def test_stop_slop_engine_rewrite() -> None:
    engine = StopSlopEngine()
    sloppy = (
        "Here's the thing: we must navigate uncertainty. "
        "This is really a game-changer. Let that sink in."
    )
    res = engine.rewrite(sloppy)

    assert "rewritten" in res
    rewritten = res["rewritten"]
    assert "Here's the thing:" not in rewritten
    assert "Let that sink in" not in rewritten
    assert "navigate" not in rewritten.lower() or "handle" in rewritten.lower()
    assert res["changes_count"] >= 3


def test_stop_slop_rules_and_examples() -> None:
    engine = StopSlopEngine()

    all_rules = engine.get_rules()
    assert "throat_clearing" in all_rules
    assert "business_jargon" in all_rules
    assert "scoring" in all_rules

    jargon_rules = engine.get_rules("business_jargon")
    assert "business_jargon" in jargon_rules
    assert "navigate" in jargon_rules["business_jargon"]["replacements"]

    examples = engine.get_examples()
    assert len(examples) == 5
    assert "before" in examples[0]
    assert "after" in examples[0]


def test_stop_slop_tool_factories() -> None:
    analyze_tool = make_stop_slop_analyze_tool()
    assert analyze_tool.name == "stop_slop_analyze"
    res_a = analyze_tool.handler({"text": "Direct statement."})
    assert isinstance(res_a, dict)
    assert res_a["passes_threshold"] is True

    rewrite_tool = make_stop_slop_rewrite_tool()
    assert rewrite_tool.name == "stop_slop_rewrite"
    res_r = rewrite_tool.handler({"text": "Here's the thing: just ship it."})
    assert isinstance(res_r, dict)
    assert "Here's the thing" not in res_r["rewritten"]

    rules_tool = make_stop_slop_rules_tool()
    assert rules_tool.name == "stop_slop_rules"
    res_rules = rules_tool.handler({"category": "scoring"})
    assert "scoring" in res_rules

    examples_tool = make_stop_slop_examples_tool()
    assert examples_tool.name == "stop_slop_examples"
    res_ex = examples_tool.handler({})
    assert len(res_ex["examples"]) == 5


def test_builtin_tools_registers_stop_slop(tmp_path: Path) -> None:
    ctx = Context()
    tools = ToolRegistry(ctx)
    ctx.provide("tools", tools)

    plugin = BuiltinToolsPlugin(workspace=tmp_path, enable_stop_slop=True)
    plugin.apply(ctx)

    names = set(tools.tools.keys())
    assert "stop_slop_analyze" in names
    assert "stop_slop_rewrite" in names
    assert "stop_slop_rules" in names
    assert "stop_slop_examples" in names
    assert "read_file" in names


def test_default_skills_has_stop_slop() -> None:
    skills = default_skills()
    slop_skill = next((s for s in skills if s.name == "stop-slop"), None)
    assert slop_skill is not None
    assert "stop_slop_analyze" in slop_skill.tools
    assert "stop_slop_rewrite" in slop_skill.tools
    assert "slop" in slop_skill.triggers
    assert "prose" in slop_skill.triggers
    assert slop_skill.min_risk == RiskLevel.LOW


def test_skill_catalog_matching_stop_slop() -> None:
    catalog = SkillCatalog(default_skills())
    matches = catalog.match("Please help de-slop and edit my prose draft")
    matched_names = [s.name for s, _ in matches]
    assert "stop-slop" in matched_names

    matches2 = catalog.match("Eliminate throat clearing, jargon, and adverbs from this text")
    matched_names2 = [s.name for s, _ in matches2]
    assert "stop-slop" in matched_names2


def test_skills_plugin_auto_discovers_installed_bundle(tmp_path: Path) -> None:
    ctx = Context()
    tools = ToolRegistry(ctx)
    ctx.provide("tools", tools)

    # When skills_dir is None, SkillsPlugin automatically searches repo/cwd skills/ directory
    plugin = SkillsPlugin(skills_dir=None)
    plugin.apply(ctx)

    catalog = cast(SkillCatalog, ctx.get("skills"))
    skill = catalog.get("stop-slop")
    assert skill is not None
    assert "stop_slop_analyze" in skill.tools


@pytest.mark.asyncio
async def test_tool_gating_unlocked_by_stop_slop_trigger(tmp_path: Path) -> None:
    ctx = Context()
    tools = ToolRegistry(ctx)
    ctx.provide("tools", tools)

    builtin = BuiltinToolsPlugin(workspace=tmp_path, enable_bash=False)
    builtin.apply(ctx)

    skills_plug = SkillsPlugin(gate_tools=True)
    skills_plug.apply(ctx)

    # 1. Non-matching objective: stop_slop tools should not be active
    req1 = {
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "tools": tools.schemas(),
    }
    out1 = await ctx.events.dispatch("agent/request", "waterfall", req1)
    tool_names_1 = [t["function"]["name"] for t in out1.get("tools", [])]
    assert "stop_slop_analyze" not in tool_names_1
    assert "read_file" in tool_names_1  # unmanaged tools remain allowed

    # 2. Matching objective: stop_slop tools unlocked
    req2 = {
        "messages": [{"role": "user", "content": "Review this prose and remove slop and buzzwords"}],
        "tools": tools.schemas(),
    }
    out2 = await ctx.events.dispatch("agent/request", "waterfall", req2)
    tool_names_2 = [t["function"]["name"] for t in out2.get("tools", [])]
    assert "stop_slop_analyze" in tool_names_2
    assert "stop_slop_rewrite" in tool_names_2


def test_cli_slop_analyze_flag(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--slop-analyze", "Technology is manageable. People aren't."])
    assert code == 0
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["passes_threshold"] is True
    assert report["total_score"] >= 45


def test_cli_slop_rewrite_flag(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--slop-rewrite", "Here's the thing: navigate the uncertainty."])
    assert code == 0
    captured = capsys.readouterr()
    assert "Here's the thing" not in captured.out
    assert "handle" in captured.out.lower() or "uncertainty" in captured.out


def test_cli_slop_rules_and_examples(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--slop-rules", "scoring"])
    assert code == 0
    captured = capsys.readouterr()
    rules = json.loads(captured.out)
    assert "scoring" in rules

    code_ex = main(["--slop-examples"])
    assert code_ex == 0
    captured_ex = capsys.readouterr()
    examples = json.loads(captured_ex.out)
    assert len(examples) == 5


@pytest.mark.asyncio
async def test_repl_slop_commands(tmp_path: Path) -> None:
    commands = [
        "/slop Here's the thing: we must lean into it.",
        "/deslop Here's the thing: we must lean into it.",
        "/exit",
    ]
    idx = 0

    def mock_input(_prompt: str = "") -> str:
        nonlocal idx
        val = commands[idx]
        idx += 1
        return val

    output_lines: list[str] = []

    def mock_output(text: str) -> None:
        output_lines.append(text)

    cfg = HarnessConfig(
        mock="ok",
        workspace_dir=tmp_path,
        session_path=tmp_path / "test_session.jsonl",
    )
    code = await run_repl(cfg, input_func=mock_input, output_func=mock_output)
    assert code == 0
    full_output = "".join(output_lines)
    assert "Violations:" in full_output
    assert "throat_clearing" in full_output
    assert "Rewritten:" in full_output
    assert "accept" in full_output or "Changes" in full_output
