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