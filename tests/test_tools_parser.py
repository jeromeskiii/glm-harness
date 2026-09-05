"""Tests for the tool-call parser."""

from __future__ import annotations

from glmharness.tools import parse_tool_calls


def test_single_call_with_string_arg() -> None:
    text = (
        '<tool_call>get_weather<arg_key>city</arg_key>'
        '<arg_value>"Paris"</arg_value></tool_call>'
    )
    assert parse_tool_calls(text) == [{"name": "get_weather", "arguments": {"city": "Paris"}}]


def test_multiple_calls_in_one_message() -> None:
    text = (
        '<tool_call>add<arg_key>a</arg_key><arg_value>1</arg_value>'
        '<arg_key>b</arg_key><arg_value>2</arg_value></tool_call>'
        '<tool_call>sub<arg_key>x</arg_key><arg_value>5</arg_value></tool_call>'
    )
    assert parse_tool_calls(text) == [
        {"name": "add", "arguments": {"a": 1, "b": 2}},
        {"name": "sub", "arguments": {"x": 5}},
    ]


def test_object_value_is_decoded() -> None:
    text = (
        '<tool_call>fn<arg_key>opts</arg_key>'
        '<arg_value>{"k": [true, false]}</arg_value></tool_call>'
    )
    assert parse_tool_calls(text) == [{"name": "fn", "arguments": {"opts": {"k": [True, False]}}}]


def test_call_without_arguments() -> None:
    text = "<tool_call>ping</tool_call>"
    assert parse_tool_calls(text) == [{"name": "ping", "arguments": {}}]


def test_undecodable_value_falls_back_to_string() -> None:
    text = "<tool_call>x<arg_key>k</arg_key><arg_value>{broken</arg_value></tool_call>"
    assert parse_tool_calls(text) == [{"name": "x", "arguments": {"k": "{broken"}}]


def test_empty_blocks_are_skipped() -> None:
    text = "<tool_call></tool_call><tool_call>ok</tool_call>"
    assert parse_tool_calls(text) == [{"name": "ok", "arguments": {}}]


def test_no_tool_calls_yields_empty_list() -> None:
    assert parse_tool_calls("nothing here") == []
