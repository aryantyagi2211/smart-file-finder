"""
tests/unit/test_command_parser.py

Basic unit tests for app/command_parser.py.
"""

import pytest
from app.command_parser import parse_command, Command


def test_find_command():
    result = parse_command("/find my resume")
    assert result.command == Command.FIND
    assert result.query == "my resume"


def test_open_command():
    result = parse_command("/open budget spreadsheet")
    assert result.command == Command.OPEN
    assert result.query == "budget spreadsheet"


def test_show_command():
    result = parse_command("/show that diagram")
    assert result.command == Command.SHOW
    assert result.query == "that diagram"


def test_plain_text_defaults_to_find():
    result = parse_command("just plain text query")
    assert result.command == Command.FIND
    assert result.query == "just plain text query"


def test_empty_input():
    result = parse_command("")
    assert result.command == Command.FIND
    assert result.query == ""


def test_unknown_command_raises():
    with pytest.raises(ValueError):
        parse_command("/unknown something")


def test_command_is_case_insensitive():
    result = parse_command("/FIND my resume")
    assert result.command == Command.FIND