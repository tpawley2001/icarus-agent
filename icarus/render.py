"""Terminal output. ANSI when it's a TTY, plain text when piped."""

from __future__ import annotations

import itertools
import shutil
import sys
import threading
import time
from typing import Optional


class Style:
    def __init__(self, color: bool = True):
        self.color = color and sys.stdout.isatty()

    def _w(self, code: str, s: str) -> str:
        return f"\033[{code}m{s}\033[0m" if self.color else s

    def dim(self, s): return self._w("2", s)
    def bold(self, s): return self._w("1", s)
    def red(self, s): return self._w("31", s)
    def green(self, s): return self._w("32", s)
    def yellow(self, s): return self._w("33", s)
    def blue(self, s): return self._w("34", s)
    def magenta(self, s): return self._w("35", s)
    def cyan(self, s): return self._w("36", s)
    def grey(self, s): return self._w("90", s)


class Spinner:
    """Progress indicator that also shows elapsed time.

    Cold-loading a 35B through llama-swap can take minutes; without a running
    clock the CLI looks hung and people kill it half a second before it works.
    """

    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, style: Style, enabled: bool = True):
        self.style = style
        self.enabled = enabled and sys.stdout.isatty()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._label = ""
        self._start = 0.0
        # Predicate; while it returns True the spinner leaves the line
        # alone so a half-typed prompt is not overwritten.
        self.suppress = lambda: False

    def _run(self) -> None:
        for frame in itertools.cycle(self.FRAMES):
            if self._stop.is_set():
                break
            if self.suppress():
                time.sleep(0.1)
                continue
            elapsed = time.time() - self._start
            clock = f"{elapsed:4.0f}s" if elapsed >= 1 else "    "
            line = f"\r{self.style.cyan(frame)} {self._label} {self.style.grey(clock)}"
            sys.stdout.write(line[: (shutil.get_terminal_size().columns - 1)])
            sys.stdout.flush()
            time.sleep(0.1)
        self.clear()

    def clear(self) -> None:
        if self.enabled and not self.suppress():
            sys.stdout.write("\r" + " " * (shutil.get_terminal_size().columns - 1) + "\r")
            sys.stdout.flush()

    def start(self, label: str) -> None:
        if not self.enabled:
            return
        self.stop()
        self._label = label
        self._start = time.time()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def update(self, label: str) -> None:
        self._label = label

    def stop(self) -> None:
        if self._thread and self._thread.is_alive():
            self._stop.set()
            self._thread.join(timeout=1)
        self._thread = None


def human_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def bar(fraction: float, width: int = 20) -> str:
    fraction = max(0.0, min(1.0, fraction))
    filled = int(fraction * width)
    return "█" * filled + "░" * (width - filled)
