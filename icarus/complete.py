"""Tab completion at the prompt: filesystem paths and slash commands.

Paths are what you actually type at an agent — "read src/main.py", "/cwd
~/projects" — and typing them out in full is the slow part of a prompt. Tab
completes them here the way a shell does, including inside a longer sentence.

Two portability facts drive the shape of this, both verified on this box:

* The interpreter may link **libedit** instead of GNU readline (python.org and
  Homebrew builds do). libedit parses GNU's ``tab: complete`` inputrc line
  without complaint and then ignores it, leaving Tab self-inserting, so the
  binding has to be spelled ``bind ^I rl_complete`` for that backend.
* Python's readline module sets ``rl_completion_append_character`` to NUL, so
  neither backend appends a space after a unique match. Completing a directory
  to ``src/`` therefore leaves the cursor ready for the next Tab, which is
  exactly what is wanted and is not something to work around.
"""

from __future__ import annotations

import os
from typing import Callable, List, Optional, Sequence

try:
    import readline
except ImportError:  # pragma: no cover — Windows without pyreadline3
    readline = None

# python.org and Homebrew builds link libedit rather than GNU readline, and
# libedit does not understand GNU's inputrc syntax: `tab: complete` parses
# without error and silently leaves Tab self-inserting. The `python3` on PATH
# here is one of those, so this is not a theoretical portability nicety.
LIBEDIT = bool(readline) and "libedit" in (getattr(readline, "__doc__", "") or "")


def split_token(line: str, beg: int, end: int) -> tuple:
    """Return ``(start, token)`` — the real word under the cursor.

    readline splits words on its delimiters, which cannot see a backslash
    before a space, so for ``docs/two\\ wo`` it hands the completer only
    ``wo``. Walking the buffer back over escaped spaces keeps a path with a
    space in it a single token.
    """
    start = beg
    while start > 0:
        if line[start - 1] in " \t":
            if start >= 2 and line[start - 2] == "\\":
                start -= 2
                continue
            break
        start -= 1
    return start, line[start:end]


def unescape(token: str) -> str:
    return token.replace("\\ ", " ")


def escape(text: str) -> str:
    return text.replace(" ", "\\ ")


def path_matches(token: str, base: str = "") -> List[str]:
    """Complete one filesystem path, preserving how the user spelled it.

    The prefix the user typed is kept verbatim in every match, so a ``~/`` stays
    a ``~/`` instead of expanding to the home directory mid-line. Relative paths
    resolve against `base` — the agent's working directory, which `/cwd` can
    move away from the process's own cwd.
    """
    raw = unescape(token)
    if raw == "~":
        return ["~/"]

    frag = raw.rpartition("/")[2]
    prefix = raw[: len(raw) - len(frag)]         # "", "~/", "/usr/", "src/"
    lookup = os.path.expanduser(prefix) if prefix else "."
    if not os.path.isabs(lookup) and base:
        lookup = os.path.join(base, lookup)

    try:
        names = os.listdir(lookup or ".")
    except OSError:
        return []

    out = []
    for name in names:
        if not name.startswith(frag):
            continue
        # Dotfiles only once the user has committed to one, as in a shell.
        if not frag.startswith(".") and name.startswith("."):
            continue
        full = prefix + name
        if os.path.isdir(os.path.join(lookup or ".", name)):
            full += "/"
        out.append(full)
    return sorted(out)


def command_matches(token: str, commands: Sequence[str]) -> List[str]:
    return sorted(c for c in commands if c.startswith(token))


class Completer:
    """Tab completion for the prompt: slash commands and filesystem paths."""

    def __init__(self, commands: Sequence[str] = (),
                 base: Optional[Callable[[], str]] = None) -> None:
        self.commands = tuple(commands)
        self.base = base
        self.matches: List[str] = []

    def candidates(self, line: str, beg: int, end: int) -> List[str]:
        start, token = split_token(line, beg, end)
        out: List[str] = []
        # A leading "/" is ambiguous — "/model" is a command, "/etc" is a path.
        # At the start of the line a command is nearly always what was meant,
        # so those win outright and paths only get a look in when no command
        # matches: "/mo" completes to /model, "/c" lists commands rather than
        # burying them under /cdrom, and "/ho" still reaches /home/.
        if start == 0 and token.startswith("/"):
            out += command_matches(token, self.commands)
            if out:
                cut = beg - start
                return [escape(m)[cut:] for m in out]
        base = ""
        if self.base is not None:
            try:
                base = str(self.base() or "")
            except Exception:
                base = ""
        for m in path_matches(token, base):
            if m not in out:
                out.append(m)
        # readline only replaces [beg, end), so hand back the tail of each
        # match; the head is the escaped text already on the line.
        cut = beg - start
        return [escape(m)[cut:] for m in out]

    def __call__(self, text: str, state: int) -> Optional[str]:
        if state == 0:
            if readline is None:
                return None
            try:
                self.matches = self.candidates(
                    readline.get_line_buffer(),
                    readline.get_begidx(),
                    readline.get_endidx(),
                )
            except Exception:
                # A completer that raises makes the whole prompt unusable.
                self.matches = []
        return self.matches[state] if state < len(self.matches) else None


def install(commands: Sequence[str] = (),
            base: Optional[Callable[[], str]] = None) -> Optional[Completer]:
    """Bind Tab. Returns the completer, or None when readline is unavailable."""
    if readline is None:
        return None
    completer = Completer(commands, base)
    readline.set_completer(completer)
    # Only whitespace ends a word: "/" and "-" and "." belong to the path.
    readline.set_completer_delims(" \t\n")
    readline.parse_and_bind("bind ^I rl_complete" if LIBEDIT else "tab: complete")
    # No set_completion_display_matches_hook: printing the ambiguous set as
    # bare names reads better, but neither backend redraws the prompt line
    # afterwards (libedit ignores the hook outright, GNU leaves the cursor
    # where the hook left it), so the rest of what you type lands on a line
    # with no prompt. Their own listing is correct; take it.
    return completer
