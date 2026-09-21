"""Tests for the verdict layer and its wiring into the tool registry."""

from __future__ import annotations

from glmharness import Consequence, Tool, ToolRegistry, VerificationRun, judge
from glmharness.artifacts.flow_step_tool import step_step_call, step_step_event
from glmharness.verdict.verified_constants import VERDICT_ERROR, VERDICT_OK


def test_judge_maps_result_ok_field() -> None:
    c = Consequence(tool="x", expect="ok", arguments={})
    assert judge(c, {"ok": True}) == VERDICT_OK
    assert judge(c, {"ok": False}) == VERDICT_ERROR


def test_judge_unknown_on_non_mapping_and_missing_ok() -> None:
    c = Consequence(tool="x", expect="ok", arguments={})
    assert judge(c, "not a mapping") == "unknown"
    assert judge(c, {"content": "no ok key"}) == "unknown"


def test_verification_run_folds_empty_and_mixed_steps() -> None:
    run = VerificationRun()
    assert run.verdict() == "unknown"
    c = Consequence(tool="x", expect="ok", arguments={})
    assert run.record(c, {"ok": True}) == VERDICT_OK
    run.record(c, {"ok": False})
    assert run.verdict() == VERDICT_ERROR


def test_flow_step_tool_step_records() -> None:
    call = step_step_call("pre-execute", {"name": "echo", "id": "1"})
    assert call == {"step": "pre-execute", "call": {"name": "echo", "id": "1"}}
    assert step_step_event("post-execute", "tools/post-execute") == {
        "step": "post-execute",
        "event": "tools/post-execute",
    }


async def test_registry_records_verdict_in_tool_result_fact(ctx, log, echo_tool) -> None:
    ctx.provide("sessions", log)
    registry = ToolRegistry(ctx)
    registry.register(echo_tool)
    result = await registry.execute("echo", {"a": 1})
    assert result["ok"] is True
    fact = log.events[-1]
    assert fact.type == "tool/result"
    assert fact.data["verdict"] == VERDICT_OK
    steps = fact.data["steps"]
    assert [s["step"] for s in steps] == ["pre-execute", "post-execute"]
    assert registry.verification_run.verdict() == VERDICT_OK


async def test_registry_records_error_verdict(ctx, log) -> None:
    ctx.provide("sessions", log)
    registry = ToolRegistry(ctx)
    denied = Tool("x", "x", {"type": "object"}, lambda a: a, allowed=False)
    registry.register(denied)
    result = await registry.execute("x", {})
    assert result["error"] == "DENIED_BY_POLICY"
    fact = log.events[-1]
    assert fact.data["verdict"] == VERDICT_ERROR
    assert registry.verification_run.verdict() == VERDICT_ERROR
