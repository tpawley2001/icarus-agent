"""Tests for prompt tab completion.

The interesting cases are the ones a naive `glob(text + "*")` gets wrong: a
path with a space in it, a `~` that must survive completion instead of
expanding mid-line, a leading `/` that could be either a slash command or an
absolute path, and relative paths that have to resolve against the agent's
working directory rather than the process's.

Stdlib only, no test runner, in keeping with the rest of the project:

    python3 tests/test_completion.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from icarus.complete import (  # noqa: E402
    Completer,
    command_matches,
    escape,
    path_matches,
    split_token,
    unescape,
)

FAILURES: list = []


def check(label: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(label)


def section(title: str) -> None:
    print(f"\n{title}")


TMP = Path(tempfile.mkdtemp(prefix="icarus-complete-"))
(TMP / "src").mkdir()
(TMP / "src" / "deep").mkdir()
(TMP / "src" / "main.py").touch()
(TMP / "two words.txt").touch()
(TMP / ".hidden").touch()
(TMP / "notes.md").touch()

CMDS = ("/caps", "/compact", "/cost", "/cwd", "/help", "/model", "/models")


def complete(line: str, base: str = str(TMP)) -> list:
    """What Tab would offer, with the cursor at the end of `line`."""
    end = len(line)
    beg = end
    while beg > 0 and line[beg - 1] not in " \t":
        beg -= 1
    return Completer(CMDS, base=lambda: base).candidates(line, beg, end)


section("[1] ordinary path completion")
check("a directory completes with a trailing slash",
      complete("read sr") == ["src/"], f"{complete('read sr')}")
check("completion continues inside the directory",
      complete("read src/ma") == ["src/main.py"])
check("a bare fragment lists every match, sorted",
      complete("n") == ["notes.md"])
check("no match yields nothing rather than an error", complete("zzz") == [])
check("a nonexistent directory is not an error", complete("nope/x") == [])

section("[2] dotfiles behave like a shell's")
check("dotfiles stay hidden until asked for", ".hidden" not in complete(""))
check("a leading dot reveals them", complete(".hi") == [".hidden"])

section("[3] a path with a space in it")
# readline's delimiters cannot see the backslash, so it hands the completer
# only "wo" and replaces only that; the match has to be the matching tail.
line = "read two\\ wo"
start, token = split_token(line, len(line) - 2, len(line))
check("the escaped space is walked back into one token", token == "two\\ wo",
      repr(token))
check("unescape/escape round-trips", escape(unescape(token)) == token)
check("only the replaced tail comes back",
      complete("read two\\ wo") == ["words.txt"],
      f"{complete('read two\\\\ wo')}")
check("the space stays escaped in a full-token completion",
      complete("read two") == ["two\\ words.txt"])

section("[4] ~ survives instead of expanding mid-line")
home = Path(os.path.expanduser("~"))
first = sorted(p.name for p in home.iterdir() if not p.name.startswith("."))[:1]
if first:
    got = complete("~/" + first[0][:2])
    check("a home-relative path keeps its ~", all(m.startswith("~/") for m in got),
          f"{got[:3]}")
check("a bare ~ offers the separator", path_matches("~") == ["~/"])

section("[5] a leading / is a command first, a path second")
check("a command wins at the start of the line",
      complete("/mod") == ["/model", "/models"], f"{complete('/mod')}")
check("commands are not buried under filesystem entries",
      all(m.startswith("/c") and not m.startswith("/cd") for m in complete("/c")),
      f"{complete('/c')}")
check("an absolute path still works when no command matches",
      complete("/hom") == ["/home/"] or complete("/hom") == [],
      f"{complete('/hom')}")
check("mid-line, a / is only ever a path",
      complete("/cwd /hom") == ["/home/"] or complete("/cwd /hom") == [],
      f"{complete('/cwd /hom')}")
check("command matching is exact-prefix", command_matches("/mo", CMDS) ==
      ["/model", "/models"])

section("[6] relative paths follow the agent's workdir, not the process's")
other = TMP / "elsewhere"
other.mkdir()
(other / "unique-name.txt").touch()
check("completion resolves against the base directory",
      Completer(CMDS, base=lambda: str(other)).candidates("uni", 0, 3)
      == ["unique-name.txt"])
check("the process cwd is not consulted", complete("uni") == [])
check("a base that raises does not break the prompt",
      Completer(CMDS, base=lambda: 1 / 0).candidates("sr", 0, 2) is not None)

section("[7] the readline protocol, driven the way readline drives it")


class FakeReadline:
    """readline calls the completer with (text, state) and reads the rest of
    the context back off the module, so the module is the interface."""

    def __init__(self, line: str) -> None:
        self.line, self.end = line, len(line)
        self.beg = self.end
        while self.beg > 0 and line[self.beg - 1] not in " \t":
            self.beg -= 1

    def get_line_buffer(self) -> str:
        return self.line

    def get_begidx(self) -> int:
        return self.beg

    def get_endidx(self) -> int:
        return self.end


def offer(line: str) -> list:
    """Every match readline would collect, by walking state 0,1,2,..."""
    import icarus.complete as mod
    saved, mod.readline = mod.readline, FakeReadline(line)
    try:
        c = Completer(CMDS, base=lambda: str(TMP))
        out, state = [], 0
        while True:
            m = c(line[mod.readline.beg:], state)
            if m is None:
                return out
            out.append(m)
            state += 1
    finally:
        mod.readline = saved


check("walking state returns every match then None",
      offer("read src/") == ["src/deep/", "src/main.py"], f"{offer('read src/')}")
check("state past the end terminates the walk", offer("read zzz") == [])
check("an absolute path needs no base", path_matches("/et")[:1] in ([], ["/etc/"]))


section("[8] Tab is bound in the dialect the backend actually speaks")
import icarus.complete as _mod  # noqa: E402

check("libedit is detected from the module docstring, not the platform",
      _mod.LIBEDIT == ("libedit" in (__import__("readline").__doc__ or "")))
check("install() is a no-op without readline rather than a crash",
      (lambda: (setattr(_mod, "readline", None),
                _mod.install(CMDS) is None)[1])() is True)
_mod.readline = __import__("readline")


shutil.rmtree(TMP, ignore_errors=True)
print("\n" + ("ALL PASS" if not FAILURES else f"{len(FAILURES)} FAILURE(S): {FAILURES}"))
sys.exit(1 if FAILURES else 0)
