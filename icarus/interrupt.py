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

---- the bug this file used to have ----

The reader thread and the main thread's ``input()`` both read the same tty fd.
``stop()`` set a flag and ``join(timeout=0.5)``-ed the thread, but a raw-mode
``os.read(fd, 1)`` is *blocking*: if a byte the thread's ``select`` had flagged
was consumed by the other reader first, the read would sit there forever,
``join`` would time out, and ``stop()`` returned having leaked a live thread.
That zombie then raced the next prompt's readline for keystrokes — some letters
landed in the watcher's buffer, some in readline's line — which is exactly the
"suddenly my sentence is missing letters" symptom, and the garbled line is what
got relayed to the model.

Three changes close the race:

  1. stdin is put in non-blocking mode while raw mode is active, so the reader
     can never block in ``os.read`` and always returns to its ``select`` loop,
     where it promptly notices ``_stop``.
  2. a self-pipe is watched alongside stdin so ``stop()`` wakes the thread
     immediately instead of waiting out a select timeout.
  3. ``stop()`` joins without giving up early, so no zombie thread can survive
     into the next prompt.
"""

from __future__ import annotations

import codecs
import os
import queue
import sys
import threading
from typing import List, Optional

try:  # POSIX only; absence just disables the feature.
    import fcntl
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
        self._saved_fl = None
        # Self-pipe: stop() writes a byte here so the reader's select wakes
        # immediately rather than waiting out its timeout.
        self._wake_r: Optional[int] = None
        self._wake_w: Optional[int] = None

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
        if not self.enabled:
            return
        if self._thread is not None and self._thread.is_alive():
            # A previous stop() must have finished its join; refuse to spawn a
            # second reader on the same fd.
            return
        fd = sys.stdin.fileno()
        try:
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
            # Non-blocking so the reader can never stall in os.read; the flag
            # is a separate fcntl state from termios and must be saved/restored
            # on its own.
            self._saved_fl = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, self._saved_fl | os.O_NONBLOCK)
        except Exception:
            # No controlling terminal after all — disable rather than fail.
            self.enabled = False
            self._saved_term = None
            self._saved_fl = None
            return
        self._wake_r, self._wake_w = os.pipe()
        os.set_blocking(self._wake_r, False)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._wake_w is not None:
            try:
                os.write(self._wake_w, b"x")
            except OSError:
                pass
        if self._thread is not None and self._thread.is_alive():
            # The reader now only ever blocks in select, which the wake byte
            # breaks immediately, so this join cannot time out in practice; a
            # generous bound is a last-ditch guard against a stuck thread.
            self._thread.join(timeout=2.0)
        self._thread = None
        self.clear_line()
        for pipe_fd in (self._wake_r, self._wake_w):
            if pipe_fd is not None:
                try:
                    os.close(pipe_fd)
                except OSError:
                    pass
        self._wake_r = self._wake_w = None
        fd = sys.stdin.fileno()
        if self._saved_fl is not None:
            try:
                fcntl.fcntl(fd, fcntl.F_SETFL, self._saved_fl)
            except Exception:
                pass
            self._saved_fl = None
        if self._saved_term is not None:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, self._saved_term)
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
        wake_r = self._wake_r
        # Decode incrementally: os.read hands us raw bytes, and a multi-byte
        # UTF-8 character may be split across reads. Feeding the decoder one
        # byte at a time lets it reassemble 'é', emoji, etc. instead of turning
        # each stray byte into mojibake or silently dropping it.
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        while not self._stop.is_set():
            try:
                ready, _, _ = select.select([sys.stdin, wake_r], [], [], 0.1)
            except Exception:
                return
            if wake_r is not None and wake_r in ready:
                try:
                    os.read(wake_r, 4096)
                except Exception:
                    pass
            if sys.stdin not in ready:
                continue
            try:
                data = os.read(fd, 4096)
            except BlockingIOError:
                continue
            except Exception:
                return
            if not data:
                continue
            for byte in data:
                ch = decoder.decode(bytes([byte]))
                if not ch:
                    continue
                self._handle(fd, ch)

    def _handle(self, fd: int, ch: str) -> None:
        if ch == "\x1b":
            # Esc is also the prefix of arrow/function keys. If more bytes are
            # already waiting it was a key sequence, not a real Esc — drain the
            # sequence rather than aborting the user's turn.
            more, _, _ = select.select([sys.stdin], [], [], 0.05)
            if more:
                try:
                    os.read(fd, 64)
                except BlockingIOError:
                    pass
                except Exception:
                    pass
                return
            self._abort.set()
            with self._lock:
                self._buf = ""
            self.clear_line()
            return

        if ch in ("\x03", "\x04"):  # Ctrl-C / Ctrl-D
            self._abort.set()
            self.clear_line()
            return

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
