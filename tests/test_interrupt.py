"""Regression test for the terminal-input race that dropped keystrokes.

The bug: the InputWatcher's raw-mode reader thread and the main thread's
``input()`` both read the same tty fd. ``stop()`` used to ``join(timeout=0.5)``
and give up if the thread was blocked in a *blocking* ``os.read(fd, 1)`` — which
happens whenever the other reader wins the race for a byte. That leaked a live
reader into the next prompt, which then stole letters from the user's typing.
Real symptom, caught in the session log: "the whole fucknews app" reached the
model as "the wo fucknews app"; "new project" arrived as "ew project".

This exercises the real reader against a pseudo-terminal and checks two things
the fix guarantees:

  1. multi-byte UTF-8 is reassembled, not mojibake'd or dropped, and
  2. stop() actually terminates the thread (no zombie to race the next prompt).

Stdlib only, in keeping with the rest of the project:

    python3 tests/test_interrupt.py
"""

from __future__ import annotations

import os
import pty
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from icarus.interrupt import InputWatcher, _HAVE_TTY  # noqa: E402


class _Style:
    def cyan(self, s): return s
    def grey(self, s): return s
    def bold(self, s): return s
    def yellow(self, s): return s


class _SlaveStdin:
    """Minimal file-like over the pty slave so select() and fileno() work."""

    def __init__(self, fd: int):
        self._fd = fd

    def fileno(self) -> int:
        return self._fd

    def isatty(self) -> bool:
        return True


FAILURES: list = []


def check(label: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(label)


def main() -> int:
    if not _HAVE_TTY:
        print("SKIP — no POSIX tty support")
        return 0

    master, slave = pty.openpty()
    real_stdin = sys.stdin
    sys.stdin = _SlaveStdin(slave)  # type: ignore[assignment]
    try:
        w = InputWatcher(_Style(), enabled=True)
        w.enabled = True  # stdout is not a tty under the harness; force it on

        # --- [1] the reader thread starts, reads, and can be stopped cleanly ---
        w.start()
        thread = w._thread
        check("reader thread spawned", thread is not None and thread.is_alive())

        # --- [2] UTF-8 round-trip: multi-byte chars survive intact ---
        os.write(master, "héllo wörld — 你好\n".encode("utf-8"))
        got = None
        for _ in range(100):  # poll briefly for the queued line
            pending = w.take_messages()
            if pending:
                got = pending[0]
                break
            time.sleep(0.02)
        check("multi-byte UTF-8 reassembled, not dropped or mojibake'd",
              got == "héllo wörld — 你好", f"got {got!r}")

        # --- [3] stop() terminates the thread — no zombie reader ---
        w.stop()
        check("reader thread is dead after stop()", not thread.is_alive())
        check("no lingering thread reference", w._thread is None)

        # --- [4] a stopped watcher can start again without double-spawning ---
        w.start()
        t2 = w._thread
        w.stop()
        check("restart spawns a fresh thread that also stops cleanly",
              t2 is not None and not t2.is_alive())

    finally:
        sys.stdin = real_stdin
        os.close(master)
        os.close(slave)

    print("\n" + ("ALL PASS" if not FAILURES else f"{len(FAILURES)} FAILURE(S): {FAILURES}"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
