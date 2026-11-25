"""
app/command_parser.py

Parses raw input-bar text into a command + query, per CLAUDE.md's UI spec:
  /open <query>   -> open the matched file directly
  /show <query>   -> reveal the matched file in File Explorer
  /find <query>   -> just report the match + explanation (default)
Plain text with no leading slash-command is treated as /find.
"""

from dataclasses import dataclass
from enum import Enum
import string
import re


class Command(Enum):
    OPEN = "open"
    SHOW = "show"
    FIND = "find"


@dataclass
class ParsedCommand:
    command: Command
    query: str


VALID_COMMANDS = {
    "/open": Command.OPEN,
    "/show": Command.SHOW,
    "/find": Command.FIND,
}

DEFAULT_VOICE_COMMAND_ALIASES = {
    "show": ["show", "sho", "so", "shoe", "shaw"],
    "open": ["open", "opan"],
    "find": ["find", "fine", "fined"],
}


def parse_command(raw_input):
    """
    Parse raw input-bar text into a ParsedCommand(command, query).
    Unknown slash-commands and empty queries are handled explicitly
    rather than silently guessed at.
    """
    raw_input = raw_input.strip()

    if not raw_input:
        return ParsedCommand(command=Command.FIND, query="")

    first_word, _, rest = raw_input.partition(" ")

    if first_word.lower() in VALID_COMMANDS:
        command = VALID_COMMANDS[first_word.lower()]
        query = rest.strip()
        return ParsedCommand(command=command, query=query)

    if first_word.startswith("/"):
        # Looks like a command but isn't recognized -- don't silently
        # swallow the slash into the search query, that would confuse
        # the user about why their search returned nothing useful.
        raise ValueError(f"Unknown command: {first_word}")

    # No command prefix at all -- plain text defaults to /find.
    return ParsedCommand(command=Command.FIND, query=raw_input)


def _edit_distance_at_most_one(word, command_word):
    if abs(len(word) - len(command_word)) > 1:
        return False
    if word == command_word:
        return True

    left = right = edits = 0
    while left < len(word) and right < len(command_word):
        if word[left] == command_word[right]:
            left += 1
            right += 1
            continue
        edits += 1
        if edits > 1:
            return False
        if len(word) > len(command_word):
            left += 1
        elif len(word) < len(command_word):
            right += 1
        else:
            left += 1
            right += 1
    if left < len(word) or right < len(command_word):
        edits += 1
    return edits <= 1


def _voice_alias_command(first_word, aliases):
    for command_word, command in (
        ("open", Command.OPEN),
        ("show", Command.SHOW),
        ("find", Command.FIND),
    ):
        configured = aliases.get(command_word, [])
        if isinstance(configured, str):
            configured = [configured]
        normalized_aliases = {
            str(alias).strip().casefold()
            for alias in configured
            if str(alias).strip()
        }
        if first_word in normalized_aliases or _edit_distance_at_most_one(
            first_word, command_word
        ):
            return command
    return None


def parse_voice_command(text, aliases=None):
    """Parse spoken commands without changing typed-input command rules."""
    raw_text = str(text or "").strip().lower()
    if not raw_text:
        return ParsedCommand(command=Command.FIND, query="")

    parts = raw_text.split(maxsplit=1)
    first_word = parts[0]
    spoken_command = first_word.strip(string.punctuation)
    exact_commands = {
        "open": Command.OPEN,
        "show": Command.SHOW,
        "find": Command.FIND,
        "search": Command.FIND,
    }
    command = exact_commands.get(spoken_command)
    has_following_text = len(parts) > 1 and bool(parts[1].strip())
    if command is None and has_following_text:
        command = _voice_alias_command(
            spoken_command,
            aliases or DEFAULT_VOICE_COMMAND_ALIASES,
        )
    if command is None:
        return ParsedCommand(command=Command.FIND, query=raw_text)

    query = parts[1] if len(parts) > 1 else ""
    query = query.lstrip(string.punctuation + string.whitespace)
    fillers = {"me", "up", "the", "my", "a", "an"}
    while query:
        query_parts = query.split(maxsplit=1)
        first_query_word = query_parts[0]
        if first_query_word.strip(string.punctuation) not in fillers:
            break
        query = query_parts[1] if len(query_parts) > 1 else ""
        query = query.lstrip(string.punctuation + string.whitespace)

    query = query.rstrip(string.punctuation + string.whitespace)

    return ParsedCommand(command=command, query=query)


def should_auto_open(command, result_action):
    """Only explicit /open on a confident single-file answer may auto-open."""
    return command == Command.OPEN and result_action == "answer"