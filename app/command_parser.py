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


def parse_voice_command(text):
    """Parse spoken commands without changing typed-input command rules."""
    raw_text = str(text or "").strip().lower()
    if not raw_text:
        return ParsedCommand(command=Command.FIND, query="")

    parts = raw_text.split(maxsplit=1)
    first_word = parts[0]
    spoken_command = first_word.strip(string.punctuation)
    command = {
        "open": Command.OPEN,
        "show": Command.SHOW,
        "find": Command.FIND,
        "search": Command.FIND,
    }.get(spoken_command)
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