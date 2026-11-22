"""
tests/unit/test_command_parser.py

Basic unit tests for app/command_parser.py.
"""

import pytest
from app.command_parser import parse_command, parse_voice_command, Command


def test_find_command():
    result = parse_command("/find my resume")
    assert result.command == Command.FIND
    assert result.query == "my resume"


@pytest.mark.parametrize(
    ("text", "command", "query"),
    [
        ("Open.", Command.OPEN, ""),
        ("open the corrective rag", Command.OPEN, "corrective rag"),
        ("Show me the roadmap", Command.SHOW, "roadmap"),
        ("find roadmap", Command.FIND, "roadmap"),
        ("search for agents", Command.FIND, "for agents"),
        ("a plain sentence", Command.FIND, "a plain sentence"),
        ("opening the file", Command.FIND, "opening the file"),
        ("showcase", Command.FIND, "showcase"),
        ("sHoW mE THE RoadMap", Command.SHOW, "roadmap"),
        ("Open, corrective rag", Command.OPEN, "corrective rag"),
        ("show up my, an file", Command.SHOW, "file"),
        ("find project the file", Command.FIND, "project the file"),
    ],
)
def test_parse_voice_command(text, command, query):
    result = parse_voice_command(text)
    assert result.command == command
    assert result.query == query


def test_voice_command_normalizes_plain_sentence_to_lowercase():
    result = parse_voice_command("Keep This Exact Query")
    assert result.command == Command.FIND
    assert result.query == "keep this exact query"


def test_typed_command_words_without_slashes_remain_plain_searches():
    typed = parse_command("open corrective rag")
    assert typed.command == Command.FIND
    assert typed.query == "open corrective rag"


def test_typed_slash_commands_keep_existing_behavior():
    assert parse_command("/open x").command == Command.OPEN
    assert parse_command("/show x").command == Command.SHOW


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