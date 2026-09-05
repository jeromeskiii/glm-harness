"""Tests for HarnessConfig loading and validation."""

from __future__ import annotations

import pytest

from glmharness import HarnessConfig
from glmharness.errors import ConfigError


def test_defaults_validate() -> None:
    HarnessConfig().validate()


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
    config = HarnessConfig.from_env()
    assert config.max_rounds == 5
    assert config.reasoning_effort == "low"
    assert config.log_format == "json"


def test_from_env_rejects_bad_values(monkeypatch) -> None:
    monkeypatch.setenv("GLMH_MAX_ROUNDS", "not-a-number")
    with pytest.raises(ConfigError):
        HarnessConfig.from_env()


def test_from_env_unknown_keys_collected(monkeypatch) -> None:
    monkeypatch.setenv("GLMH_TYPO", "x")
    monkeypatch.setenv("GLMH_LOG_FROMAT", "json")  # note the typo
    config = HarnessConfig.from_env()
    assert "GLMH_TYPO" in config.unknown_env_keys()
    assert "GLMH_LOG_FROMAT" in config.unknown_env_keys()


def test_request_timeout_zero_disabled_is_valid() -> None:
    HarnessConfig(request_timeout_s=0).validate()
