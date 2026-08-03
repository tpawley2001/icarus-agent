"""Type-ahead and interrupt while the model is working.

Same affordance hermes and Claude Code give you: the turn is not a dead end.
While the model streams or a tool runs you can

  * type a line and press Enter — it is queued and injected at the next
    iteration boundary as steering, without throwing away the work so far, or
  * press Esc — the in-flight request is abandoned immediately.

Implemented with a raw-mode reader on a background thread. The typed buffer
lives on a status line pinned to the bottom of the screen; every other writer
goes through ``Console``, which clears that line first and redraws it after, so
streamed tokens and your half-typed prompt never scribble over each other.

Degrades to a no-op when stdin or stdout is not a terminal, which is what makes
one-shot and piped mode safe.
"""

from __future__ import annotations

import os
import queue
import sys
import threading
from typing import List, Optional

try:  # POSIX only; absence just disables the feature.
    import select
    import termios
    import tty

    _HAVE_TTY = True
except ImportError:  # pragma: no cover
    _HAVE_TTY = False


class InputWatcher:
    def __init__(self, style, enabled: bool = True):
        self.style = style
        self.enabled = bool(
            enabled
            and _HAVE_TTY
            and sys.stdin is not None
            and sys.stdin.isatty()
            and sys.stdout.isatty()
        )
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._abort = threading.Event()
        self._stop = threading.Event()
        self._buf = ""
        self._shown = False
        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._saved_term = None

    # ---- state the agent loop reads ------------------------------------
    def aborted(self) -> bool:
        return self._abort.is_set()

    def take_messages(self) -> List[str]:
        out: List[str] = []
        while True:
            try:
                out.append(self._queue.get_nowait())
            except queue.Empty:
                return out

    def has_pending(self) -> bool:
        return not self._queue.empty()

    def reset(self) -> None:
        self._abort.clear()
        with self._lock:
            self._buf = ""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    # ---- the pinned status line ----------------------------------------
    def clear_line(self) -> None:
        """Erase the typed-input line so another writer can use the terminal."""
        if not self.enabled:
            return
        with self._lock:
            if self._shown:
                sys.stdout.write("\r\033[K")
                sys.stdout.flush()
                self._shown = False

    def render(self) -> None:
        """Redraw the typed-input line at the bottom of the screen."""
        if not self.enabled:
            return
        with self._lock:
            if self._buf:
                sys.stdout.write("\r\033[K" + self.style.cyan("› ") + self._buf)
                sys.stdout.flush()
                self._shown = True

    # ---- lifecycle -------------------------------------------------------
    def start(self) -> None:
        if not self.enabled or (self._thread and self._thread.is_alive()):
            return
        try:
            fd = sys.stdin.fileno()
            self._saved_term = termios.tcgetattr(fd)
            mode = termios.tcgetattr(fd)
            # Clear ICANON so keys arrive immediately, and ECHO so the tty does
            # not draw them too — this renderer owns the input line, and
            # leaving ECHO on double-prints every character.
            # ISIG stays on deliberately: Ctrl-C should still raise SIGINT.
            mode[3] &= ~(termios.ICANON | termios.ECHO)
            mode[6][termios.VMIN] = 1
            mode[6][termios.VTIME] = 0
            termios.tcsetattr(fd, termios.TCSADRAIN, mode)
        except Exception:
            # No controlling terminal after all — disable rather than fail.
            self.enabled = False
            self._saved_term = None
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        self._thread = None
        self.clear_line()
        if self._saved_term is not None:
            try:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._saved_term)
            except Exception:
                pass
            self._saved_term = None

    def __enter__(self) -> "InputWatcher":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # ---- reader ---------------------------------------------------------
    def _run(self) -> None:
        fd = sys.stdin.fileno()
        while not self._stop.is_set():
            try:
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            except Exception:
                return
            if not ready:
                continue
            try:
                ch = os.read(fd, 1).decode("utf-8", "ignore")
            except Exception:
                return
            if not ch:
                continue

            if ch == "\x1b":
                # Esc is also the prefix of arrow/function keys. If more bytes
                # are already waiting it was a key sequence, not a real Esc —
                # drain and ignore it rather than aborting the user's turn.
                more, _, _ = select.select([sys.stdin], [], [], 0.05)
                if more:
                    try:
                        os.read(fd, 8)
                    except Exception:
                        pass
                    continue
                self._abort.set()
                with self._lock:
                    self._buf = ""
                self.clear_line()
                continue

            if ch in ("\x03", "\x04"):  # Ctrl-C / Ctrl-D
                self._abort.set()
                self.clear_line()
                continue

            with self._lock:
                if ch in ("\r", "\n"):
                    text = self._buf.strip()
                    self._buf = ""
                    self.clear_line()
                    if text:
                        self._queue.put(text)
                        sys.stdout.write(
                            self.style.grey(f"  ↵ queued: {text[:70]}") + "\n"
                        )
                        sys.stdout.flush()
                elif ch in ("\x7f", "\b"):
                    self._buf = self._buf[:-1]
                    if self._buf:
                        self.render()
                    else:
                        self.clear_line()
                elif ch >= " ":
                    self._buf += ch
                    self.render()


class NullWatcher:
    """Stand-in for non-interactive runs; every method is inert."""

    enabled = False

    def aborted(self) -> bool: return False
    def take_messages(self) -> List[str]: return []
    def has_pending(self) -> bool: return False
    def reset(self) -> None: pass
    def clear_line(self) -> None: pass
    def render(self) -> None: pass
    def start(self) -> None: pass
    def stop(self) -> None: pass
    def __enter__(self): return self
    def __exit__(self, *exc): pass
