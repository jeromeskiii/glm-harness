"""Tests for the logging module."""

from __future__ import annotations

import json

import pytest

from glmharness import logging as logging_mod


def test_configure_logging_text_writes_human_to_stderr(capsys) -> None:
    logger = logging_mod.configure_logging(fmt="text", level="INFO")
    logger.info("hello", extra={"key": "value"})
    err = capsys.readouterr().err
    assert "INFO glmharness: hello" in err


def test_configure_logging_json_writes_structured(capsys) -> None:
    logger = logging_mod.configure_logging(fmt="json", level="INFO")
    logger.info("hello", extra={"round": 3, "attempt": 1})
    line = capsys.readouterr().err.strip().splitlines()[-1]
    record = json.loads(line)
    assert record["message"] == "hello"
    assert record["round"] == 3
    assert record["attempt"] == 1
    assert record["logger"] == "glmharness"


def test_configure_rejects_unknown_format() -> None:
    with pytest.raises(ValueError, match="unknown log format"):
        logging_mod.configure_logging(fmt="yaml")
