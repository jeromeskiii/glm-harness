"""Tests for Dynamic Skill Trigger & Tool Gating System."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from glmharness import (
    Context,
    RiskLevel,
    SessionLog,
    Skill,
    SkillCatalog,
    SkillsPlugin,
    Tool,
    ToolRegistry,
    import_skills_from_dir,
)
from glmharness.config import HarnessConfig
from glmharness.server import ProtocolServer


def test_tier1_exact_token_match() -> None:
    catalog = SkillCatalog(
        [
            Skill(
                name="debugger",
                triggers=frozenset({"debug"}),
            )
        ]
    )
    matches = catalog.match("Please debug this worker process.")
    assert len(matches) == 1
    assert matches[0][0].name == "debugger"
    assert matches[0][1] >= 0.65


def test_tier2_morphological_stemming_and_irregular() -> None:
    catalog = SkillCatalog(
        [
            Skill(
                name="profiler",
                triggers=frozenset({"profile"}),
            ),
            Skill(
                name="runner",
                triggers=frozenset({"run"}),
            ),
            Skill(
                name="writer",
                triggers=frozenset({"write"}),
            ),
        ]
    )
    # profiling -> profile (silent -e restoration)
    assert any(s.name == "profiler" for s, _ in catalog.match("We are profiling memory usage."))
    # ran -> run (irregular form)
    assert any(s.name == "runner" for s, _ in catalog.match("The job ran out of memory."))
    # wrote -> write (irregular form)
    assert any(s.name == "writer" for s, _ in catalog.match("Someone wrote unexpected files."))


def test_tier3_phrase_and_hyphenated_triggers() -> None:
    catalog = SkillCatalog(
        [
            Skill(
                name="code-reviewer",
                triggers=frozenset({"code review", "pull-request"}),
            )
        ]
    )
    assert any(s.name == "code-reviewer" for s, _ in catalog.match("Conduct a thorough code review please."))
    assert any(s.name == "code-reviewer" for s, _ in catalog.match("Please inspect this pull-request now."))


def test_anti_misfire_word_boundary_safety() -> None:
    catalog = SkillCatalog(
        [
            Skill(name="git-repo", triggers=frozenset({"repo"})),
            Skill(name="reader", triggers=frozenset({"read"})),
            Skill(name="schema-migrator", triggers=frozenset({"schema"})),
        ]
    )
    # "report" should NOT trigger "repo"
    assert not any(s.name == "git-repo" for s, _ in catalog.match("Generate a summary report."))
    # "already" or "thread" should NOT trigger "read"
    assert not any(s.name == "reader" for s, _ in catalog.match("The thread has already terminated."))
    # "schematic" should NOT trigger "schema"
    matches = catalog.match("The schematic diagram was updated.")
    assert not any(s.name == "schema-migrator" for s, _ in matches)


def test_multi_signal_scoring() -> None:
    catalog = SkillCatalog(
        [
            Skill(
                name="always-on",
                always=True,
            ),
            Skill(
                name="cap-skill",
                capabilities=frozenset({"diagnostics", "security"}),
            ),
            Skill(
                name="multi-hit",
                triggers=frozenset({"bug", "crash", "trace"}),
            ),
        ]
    )
    # Always on
    matches = catalog.match("Unrelated task prompt")
    assert any(s.name == "always-on" and score == 1.0 for s, score in matches)

    # Capability matching
    cap_matches = catalog.match(
        "Task prompt",
        capabilities=frozenset({"diagnostics"}),
    )
    assert any(s.name == "cap-skill" and score > 0.7 for s, score in cap_matches)

    # Multi-trigger hit ramp
    ramp_matches = catalog.match("Investigating a bug and crash with a trace")
    multi_hit = next(s for s, _ in ramp_matches if s.name == "multi-hit")
    score = next(sc for s, sc in ramp_matches if s.name == "multi-hit")
    assert multi_hit is not None
    # 0.45 + 0.20 * 3 = 1.05 -> clamped to 1.0
    assert score == 1.0


def test_risk_interval_gating() -> None:
    catalog = SkillCatalog(
        [
            Skill(
                name="dangerous-tool",
                triggers=frozenset({"exploit"}),
                min_risk=RiskLevel.HIGH,
                max_risk=RiskLevel.CRITICAL,
            ),
            Skill(
                name="safe-tool",
                triggers=frozenset({"inspect"}),
                min_risk=RiskLevel.LOW,
                max_risk=RiskLevel.MEDIUM,
            ),
        ]
    )
    # LOW risk task cannot trigger dangerous-tool even with trigger hit
    matches_low = catalog.match("Run exploit test", risk=RiskLevel.LOW)
    assert not any(s.name == "dangerous-tool" for s, _ in matches_low)

    # HIGH risk task unlocks dangerous-tool
    matches_high = catalog.match("Run exploit test", risk=RiskLevel.HIGH)
    assert any(s.name == "dangerous-tool" for s, _ in matches_high)

    # CRITICAL risk task blocks safe-tool (exceeds max_risk MEDIUM)
    matches_crit = catalog.match("Run inspect check", risk=RiskLevel.CRITICAL)
    assert not any(s.name == "safe-tool" for s, _ in matches_crit)


def test_import_skills_from_dir(tmp_path: Path) -> None:
    skill_dir = tmp_path / "custom-skill"
    skill_dir.mkdir()
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        """---
name: custom-auditor
description: Specialized static code auditor
tools:
  - run_linter
triggers:
  - lint
  - static-audit
category: review
tags:
  - linting
min_risk: LOW
max_risk: HIGH
---
# Instructions
Always verify clean syntax and lack of dead code.
""",
        encoding="utf-8",
    )

    imported = import_skills_from_dir(tmp_path)
    assert len(imported) == 1
    skill, valid = imported[0]
    assert valid is True
    assert skill.name == "custom-auditor"
    assert "run_linter" in skill.tools
    assert "lint" in skill.triggers
    assert "static-audit" in skill.triggers
    assert "Always verify clean syntax" in skill.guidance


def test_import_skills_from_dir_is_recursive(tmp_path: Path) -> None:
    nested = tmp_path / "vendor" / "review" / "nested-skill"
    nested.mkdir(parents=True)
    (nested / "SKILL.md").write_text(
        "---\nname: nested-auditor\ndescription: Nested auditor\n---\nInspect deeply.\n",
        encoding="utf-8",
    )
    hidden = tmp_path / ".git" / "ignored-skill"
    hidden.mkdir(parents=True)
    (hidden / "SKILL.md").write_text(
        "---\nname: hidden\ndescription: Hidden skill\n---\nDo not load.\n",
        encoding="utf-8",
    )

    imported = import_skills_from_dir(tmp_path)

    assert [(skill.name, valid) for skill, valid in imported] == [("nested-auditor", True)]


async def test_skills_plugin_waterfall_and_tool_gating(ctx: Context) -> None:
    log = SessionLog()
    ctx.provide("sessions", log)

    catalog = SkillCatalog(
        [
            Skill(
                name="bash-runner",
                tools=frozenset({"bash"}),
                triggers=frozenset({"bash", "exec"}),
                guidance="Execute commands with care.",
            ),
            Skill(
                name="db-migrator",
                tools=frozenset({"run_sql"}),
                triggers=frozenset({"migrate", "sql"}),
            ),
        ],
        gate_tools=True,
    )

    plugin = SkillsPlugin(catalog=catalog, gate_tools=True)
    plugin.apply(ctx)

    tools_registry = ToolRegistry(ctx)
    tools_registry.register(Tool("read_file", "Read file", {"type": "object"}, lambda a: a))
    tools_registry.register(Tool("bash", "Run bash", {"type": "object"}, lambda a: a))
    tools_registry.register(Tool("run_sql", "Run sql", {"type": "object"}, lambda a: a))

    # Scenario 1: Request with bash trigger
    request: dict[str, Any] = {
        "messages": [{"role": "user", "content": "Please exec a shell command"}],
        "tools": tools_registry.schemas(),
    }
    filtered_request = await ctx.events.dispatch("agent/request", "waterfall", request)

    # Verification: Guidance injected
    messages = filtered_request["messages"]
    assert any("Execute commands with care." in m["content"] for m in messages)

    # Verification: Tool schemas filtered
    active_tool_names = [t["function"]["name"] for t in filtered_request["tools"]]
    assert "read_file" in active_tool_names
    assert "bash" in active_tool_names
    assert "run_sql" not in active_tool_names  # locked out!

    # Verification: Execution gating
    # bash is allowed
    bash_res = await tools_registry.execute("bash", {"cmd": "ls"})
    assert bash_res["ok"] is True

    # run_sql is denied by skill policy
    sql_res = await tools_registry.execute("run_sql", {"q": "SELECT 1"})
    assert sql_res["ok"] is False
    assert "DENIED_BY_SKILL_POLICY" in sql_res["error"]


async def test_server_skills_rpc(tmp_path: Path) -> None:
    config = HarnessConfig(mock="dummy answer", workspace_dir=tmp_path, rpc_auto_token=False)
    server = ProtocolServer(config)
    await server.initialize_runtime()

    # Test skills/list
    list_frame = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    responses: list[dict[str, Any]] = []
    server._write_frame = lambda frame: responses.append(frame)  # type: ignore[method-assign]

    await server.handle_request(list_frame)
    assert len(responses) == 1
    assert responses[0]["result"]["abiVersion"] == 2

    responses.clear()
    list_frame = {"jsonrpc": "2.0", "id": 1, "method": "skills/list", "params": {}}
    await server.handle_request(list_frame)
    assert len(responses) == 1
    assert "skills" in responses[0]["result"]
    skill_names = [s["name"] for s in responses[0]["result"]["skills"]]
    assert "shell-command" in skill_names

    # Test skills/match
    responses.clear()
    match_frame = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "skills/match",
        "params": {"objective": "We need to debug a deadlock in worker"},
    }
    await server.handle_request(match_frame)
    assert len(responses) == 1
    matches = responses[0]["result"]["matches"]
    assert any(m["name"] == "code-debugger" for m in matches)
