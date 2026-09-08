"""Tests for HarnessConfig loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from glmharness import HarnessConfig
from glmharness.errors import ConfigError


def test_defaults_validate() -> None:
    config = HarnessConfig()
    config.validate()
    assert config.request_timeout_s == 300.0
    assert config.tool_timeout_s == 30.0
    assert config.sandbox_mode == "deny"


def test_reasoning_effort_must_be_low_high_max() -> None:
    with pytest.raises(ConfigError, match="reasoning_effort"):
        HarnessConfig(reasoning_effort="medium").validate()


def test_max_new_tokens_positive() -> None:
    with pytest.raises(ConfigError, match="max_new_tokens"):
        HarnessConfig(max_new_tokens=0).validate()


def test_max_retries_non_negative() -> None:
    with pytest.raises(ConfigError, match="max_retries"):
        HarnessConfig(max_retries=-1).validate()


def test_retry_jitter_range() -> None:
    with pytest.raises(ConfigError, match="retry_jitter"):
        HarnessConfig(retry_jitter=1.0).validate()


def test_retry_base_delay_must_be_positive_and_capped() -> None:
    with pytest.raises(ConfigError, match="retry_base_delay_s"):
        HarnessConfig(retry_base_delay_s=0).validate()
    with pytest.raises(ConfigError, match="retry_base_delay_s"):
        HarnessConfig(retry_base_delay_s=10, retry_max_delay_s=1).validate()


def test_corrupt_policy_allowed_values() -> None:
    HarnessConfig(corrupt_policy="rename").validate()
    with pytest.raises(ConfigError, match="corrupt_policy"):
        HarnessConfig(corrupt_policy="ignore").validate()


def test_model_path_must_exist_if_set(tmp_path) -> None:
    with pytest.raises(ConfigError, match="model path is not a directory"):
        HarnessConfig(model_path=tmp_path / "missing").validate()


def test_retry_delay_grows_then_caps() -> None:
    config = HarnessConfig(retry_base_delay_s=1.0, retry_max_delay_s=4.0, retry_jitter=0)
    assert config.retry_delay(1) == 1.0
    assert config.retry_delay(2) == 2.0
    assert config.retry_delay(3) == 4.0
    assert config.retry_delay(10) == 4.0


def test_from_env_reads_known_vars(monkeypatch) -> None:
    monkeypatch.setenv("GLMH_MAX_ROUNDS", "5")
    monkeypatch.setenv("GLMH_REASONING_EFFORT", "low")
    monkeypatch.setenv("GLMH_LOG_FORMAT", "json")
    monkeypatch.setenv("GLMH_MOCK", "hello")
    monkeypatch.setenv("GLMH_TOOL_ALLOWLIST", "echo, search")
    config = HarnessConfig.from_env()
    assert config.max_rounds == 5
    assert config.reasoning_effort == "low"
    assert config.log_format == "json"
    assert config.mock == "hello"
    assert config.tool_allowlist == ("echo", "search")


def test_from_env_rejects_bad_values(monkeypatch) -> None:
    monkeypatch.setenv("GLMH_MAX_ROUNDS", "not-a-number")
    with pytest.raises(ConfigError):
        HarnessConfig.from_env()


def test_from_env_rejects_invalid_boolean(monkeypatch) -> None:
    monkeypatch.setenv("GLMH_SKILL_GATE_TOOLS", "definitely")
    with pytest.raises(ConfigError, match="GLMH_SKILL_GATE_TOOLS must be bool"):
        HarnessConfig.from_env()


def test_from_env_accepts_boolean_spellings(monkeypatch) -> None:
    monkeypatch.setenv("GLMH_SKILL_GATE_TOOLS", " YES ")
    assert HarnessConfig.from_env().skill_gate_tools is True
    monkeypatch.setenv("GLMH_SKILL_GATE_TOOLS", "off")
    assert HarnessConfig.from_env().skill_gate_tools is False


def test_from_env_unknown_keys_collected(monkeypatch) -> None:
    monkeypatch.setenv("GLMH_TYPO", "x")
    monkeypatch.setenv("GLMH_LOG_FROMAT", "json")  # note the typo
    config = HarnessConfig.from_env()
    assert "GLMH_TYPO" in config.unknown_env_keys()
    assert "GLMH_LOG_FROMAT" in config.unknown_env_keys()


def test_request_timeout_zero_disabled_is_valid() -> None:
    HarnessConfig(request_timeout_s=0).validate()


def test_looks_like_snapshot(tmp_path) -> None:
    from glmharness.config import looks_like_snapshot, resolve_model_path

    assert looks_like_snapshot(tmp_path) is False
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "tokenizer_config.json").write_text("{}")
    assert looks_like_snapshot(tmp_path) is True
    config = HarnessConfig()
    resolve_model_path(config, cwd=tmp_path)
    assert config.model_path == tmp_path


def test_resolve_model_path_keeps_mock(tmp_path) -> None:
    from glmharness.config import resolve_model_path
    from glmharness.errors import ConfigError

    config = HarnessConfig(mock="hi")
    resolve_model_path(config, cwd=tmp_path)
    assert config.model_path is None
    with pytest.raises(ConfigError, match="cwd is not a GLM snapshot"):
        resolve_model_path(HarnessConfig(), cwd=tmp_path)


def test_config_sandbox_mode_validation(tmp_path: Path) -> None:
    HarnessConfig(sandbox_mode="allow").validate()
    HarnessConfig(sandbox_mode="deny").validate()
    HarnessConfig(sandbox_mode="ask").validate()
    with pytest.raises(ConfigError, match="sandbox_mode must be allow"):
        HarnessConfig(sandbox_mode="invalid").validate()


def test_config_workspace_dir_validation(tmp_path: Path) -> None:
    HarnessConfig(workspace_dir=tmp_path).validate()
    with pytest.raises(ConfigError, match="workspace path is not a directory"):
        HarnessConfig(workspace_dir=tmp_path / "non_existent_folder").validate()


def test_from_env_workspace_and_sandbox(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GLMH_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("GLMH_SANDBOX", "deny")
    config = HarnessConfig.from_env()
    assert config.workspace_dir == tmp_path
    assert config.sandbox_mode == "deny"


def test_from_env_api_base_and_key(monkeypatch) -> None:
    monkeypatch.setenv("GLMH_API_BASE", "http://localhost:8000/v1")
    monkeypatch.setenv("GLMH_API_KEY", "secret-token")
    monkeypatch.setenv("GLMH_MODEL", "my-glm-flash")
    config = HarnessConfig.from_env()
    assert config.api_base == "http://localhost:8000/v1"
    assert config.api_key == "secret-token"
    assert config.model_name == "my-glm-flash"


def test_resolve_model_path_keeps_api_base(tmp_path: Path) -> None:
    from glmharness.config import resolve_model_path

    config = HarnessConfig(api_base="http://localhost:8000/v1")
    resolve_model_path(config, cwd=tmp_path)
    assert config.model_path is None
